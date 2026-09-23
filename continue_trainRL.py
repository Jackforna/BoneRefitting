import torch
import os
import numpy as np  
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src2.GraphEncoder import GraphEncoder

from src2.dataset import BoneFragmentsDataset
from env import ArchaeologicalIncastroEnv
from callbacks import SalvaGrafico3DCallback, BaseCallback

class SalvaMigliorModelloChamfer(BaseCallback):
    def __init__(self, chamfer_da_battere=23.0, normal_da_battere=-0.06, percorso_salvataggio="ppo_archaeological_aligner", callback_grafico=None, verbose=1):
        super().__init__(verbose)
        self.best_chamfer = chamfer_da_battere
        self.best_normal = normal_da_battere
        self.percorso_salvataggio = percorso_salvataggio
        self.callback_grafico = callback_grafico  
    
    def _init_callback(self) -> None:
        if self.callback_grafico is not None:
            self.callback_grafico.init_callback(self.model)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if "chamfer" in info:
                current_chamfer = info["chamfer"]
                current_normal = info["normal_alignment"]
                
                if current_normal > self.best_normal and current_chamfer < self.best_chamfer:
                    print(f"\n🏆 NUOVO RECORD DI INCASTRO! Chamfer scesa da {self.best_chamfer:.4f} a {current_chamfer:.4f}")
                    self.best_chamfer = current_chamfer
                    print(f"\n🏆 NUOVO RECORD DI DIREZIONE! Chamfer scesa da {self.best_normal:.4f} a {current_normal:.4f}")
                    self.best_normal = current_normal

                    # 1. Salva il modello .zip
                    percorso_completo = f"{self.percorso_salvataggio}.zip"
                    self.model.save(percorso_completo)
                    print(f"💾 Modello migliorato salvato in: {percorso_completo}")
                    
                    # 2. Genera il grafico PNG classico
                    if self.callback_grafico is not None:
                        self.callback_grafico.num_timesteps = self.num_timesteps
                        self.callback_grafico._genera_e_salva_render()
                        print("📊 Grafico 3D statico (.png) salvato.")
                        
        return True

def main():
    cartella_elaborati = "data/processed/train"
    dataset_reale = BoneFragmentsDataset(
        processed_dir=cartella_elaborati, 
        num_samples=2000,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = GraphEncoder(in_channels=6).to(device)  # Inizializza il tuo GraphEncoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    env = ArchaeologicalIncastroEnv(dataset=dataset_reale, encoder=encoder, max_steps=120)
    check_env(env, warn=True)

    nome_modello = "ppo_archaeological_aligner"
    
    if os.path.exists(f"{nome_modello}.zip"):
        print(f"🔄 Trovato modello precedente. Caricamento di {nome_modello}.zip in corso...")
        model = PPO.load(nome_modello, env=env, tensorboard_log="./ppo_incastro_tensorboard/")
    else:
        print(f"Errore: Il file {nome_modello}.zip non è stato trovato.")
        return

    callback_grafico = SalvaGrafico3DCallback(cartella_output="render_training")
    
    callback_selettivo = SalvaMigliorModelloChamfer(
        chamfer_da_battere=34, 
        normal_da_battere=0.01,
        percorso_salvataggio=nome_modello,
        callback_grafico=callback_grafico
    )

    print("🚀 Ripresa addestramento con salvataggi selettivi (Modello + Grafico 3D)...")
    
    model.learn(
        total_timesteps=20000, 
        callback=callback_selettivo, 
        progress_bar=True, 
        reset_num_timesteps=False
    )

    print("Sessione di addestramento terminata.")

if __name__ == "__main__":
    main()