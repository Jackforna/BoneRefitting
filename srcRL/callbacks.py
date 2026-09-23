import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class SalvaGrafico3DCallback(BaseCallback):
    def __init__(self, save_dir="renders_3d", initial_limit=10000, initial_freq=1000, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.initial_limit = initial_limit
        self.initial_freq = initial_freq
        
        self.best_chamfer = float('inf')
        self.best_norm = -1.0
        self.best_frag_a = None
        self.best_frag_b = None
        self.best_surface = 0.0
        
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        try:
            step = self.num_timesteps
            
            # Estrazione sicura dell'ambiente reale unwrapped
            if hasattr(self.training_env, "envs"):
                env = self.training_env.envs[0].unwrapped
            else:
                env = self.training_env.unwrapped

            # --- FASE 1: Salvataggi iniziali periodici (Primi 10.000 step) ---
            if step <= self.initial_limit and step % self.initial_freq == 0:
                if env.frag_a is not None and env.frag_b is not None:
                    pts_a = env.frag_a.detach().cpu().numpy()[:, :3]
                    pts_b = env.frag_b.detach().cpu().numpy()[:, :3]
                    info = self.locals["infos"][0] if "infos" in self.locals and len(self.locals["infos"]) > 0 else {}
                    c = info.get("chamfer", 0.0)
                    n = info.get("normal_alignment", 0.0)
                    s = info.get("surface_matching_ratio", 0.0)
                    self._save_3d_snapshot(step, tag="initial", pts_a=pts_a, pts_b=pts_b, chamfer=c, normal=n, surface=s)

            # --- FASE 2: Sincronizzazione Record Globale con l'Ambiente (Normale >= 0.9) ---
            if hasattr(env, "best_frag_a") and env.best_frag_a is not None:
                env_best_c = getattr(env, "global_best_chamfer", float("inf"))
                env_best_n = getattr(env, "global_best_normal", 0.0)
                env_best_s = getattr(env, "global_best_surface_matching_ratio", 0.0)

                # Scatterà SOLO se l'ambiente ha trovato un nuovo record con Normale >= 0.9
                if env_best_s > self.best_surface and env_best_n >= 0.65:
                    self.best_chamfer = env_best_c
                    self.best_norm = env_best_n
                    self.best_surface = env_best_s
                    self.best_frag_a = env.best_frag_a.copy()
                    self.best_frag_b = env.best_frag_b.copy()

                    print(
                        f"\n🎉 [CALLBACK SYNC] Nuovo Record 2D! "
                        f"Chamfer: {self.best_chamfer:.4f} mm | Normale: {self.best_norm:.4f} | Passo: {step} | Surface: {self.best_surface}"
                    )
                    
                    # Disegniamo la VERA mesh del record salvato
                    self._save_3d_snapshot(
                        step, 
                        tag="best_attempt", 
                        pts_a=self.best_frag_a, 
                        pts_b=self.best_frag_b, 
                        chamfer=self.best_chamfer, 
                        normal=self.best_norm,
                        surface=self.best_surface
                    )

        except Exception as e:
            print(f"⚠️ Attenzione: Eccezione intercettata nel callback al passo {self.num_timesteps}: {e}")

        return True

    def _save_3d_snapshot(self, step, tag="frame", pts_a=None, pts_b=None, chamfer=0.0, normal=0.0, surface=0.0):
        try:
            if pts_a is None or pts_b is None:
                return

            pts_a_3d = pts_a[:, :3]
            pts_b_3d = pts_b[:, :3]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(pts_a_3d[:, 0], pts_a_3d[:, 1], pts_a_3d[:, 2], c='gold', s=1, label='Frammento A (Fisso)')
            ax.scatter(pts_b_3d[:, 0], pts_b_3d[:, 1], pts_b_3d[:, 2], c='deepskyblue', s=1, label='Frammento B (Mobile)')

            # Autoscale dinamico degli assi per mantenere proporzioni 1:1:1
            all_pts = np.vstack([pts_a_3d, pts_b_3d])
            max_range = np.array([
                all_pts[:, 0].max() - all_pts[:, 0].min(),
                all_pts[:, 1].max() - all_pts[:, 1].min(),
                all_pts[:, 2].max() - all_pts[:, 2].min()
            ]).max() / 2.0

            mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) * 0.5
            mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) * 0.5
            mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            ax.set_title(f"Step: {step} ({tag}) | Chamfer: {chamfer:.2f} mm | Norm: {normal:.2f} | Surface: {surface:.2f}")
            ax.legend()

            file_path = os.path.join(self.save_dir, f"{tag}_step_{step}.png")
            plt.savefig(file_path)
            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Errore durante il salvataggio dello snapshot 3D allo step {step}: {e}")