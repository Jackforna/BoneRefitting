import os
from stable_baselines3 import PPO
from env import ArchaeologicalIncastroEnv
from src.dataset import BoneFragmentsDataset
from callbacks import SalvaGrafico3DCallback

def main():
    # 1. Configura dataset e ambiente
    dataset = BoneFragmentsDataset(processed_dir="data/processed/train", num_samples=2000)
    env = ArchaeologicalIncastroEnv(dataset=dataset, max_steps=120)
    
    nome_modello = "ppo_archaeological_aligner"
    if not os.path.exists(f"{nome_modello}.zip"):
        print(f"❌ Errore: Il file {nome_modello}.zip non esiste!")
        return
        
    # 2. Carica il modello e associa l'ambiente
    model = PPO.load(nome_modello, env=env)
    
    # 3. Inizializza il tuo callback grafico originale
    callback_grafico = SalvaGrafico3DCallback(cartella_output="render_best_attempt")
    callback_grafico.init_callback(model) # Questo imposta internamente self.training_env
    
    # 4. Fai giocare il modello su un episodio
    vec_env = model.get_env()
    obs = vec_env.reset()
    
    done = False
    print("🕹️ Il modello sta muovendo i frammenti (Generazione tentativo attuale)...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        done = dones[0]
        
    # 5. ABBATTIAMO I FILTRI E FORZIAMO IL TUO RENDERING MATPLOTLIB
    print("📸 Episodio finito. Forzatura del rendering Matplotlib in corso...")
    
    # Chiamiamo DIRETTAMENTE la tua funzione che contiene la logica grafica
    callback_grafico._genera_e_salva_render()
    
    print("\n✨ Controlla ora nella cartella 'render_best_attempt'!")

if __name__ == "__main__":
    main()