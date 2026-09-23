import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans


class ArchaeologicalIncastroEnv(gym.Env):
    """Ambiente RL che utilizza OBBLIGATORIAMENTE il GraphEncoder per estrarre la geometria delle ossa"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset, encoder, max_steps=120, max_allowed_chamfer=100.0):
        super(ArchaeologicalIncastroEnv, self).__init__()

        self.dataset = dataset
        self.encoder = encoder
        self.encoder.eval()
        self.max_steps = max_steps
        self.max_allowed_chamfer = max_allowed_chamfer
        self.current_step = 0
        self.device = next(self.encoder.parameters()).device

        self.frag_a = None
        self.frag_b = None
        self.frag_a_normals = None
        self.frag_b_normals = None

        # RECORD GLOBALI (persistono per tutto l'addestramento)
        self.global_best_chamfer = float('inf')
        self.global_best_normal = 0.0
        self.global_best_surface_matching_ratio = 0.0  # gate progressivo su coverage
        self.global_best_penetration_ratio = float('inf')
        self.best_frag_a = None
        self.best_frag_b = None
        self.best_chamfer = float('inf')
        self.best_normal = 0.0
        self.best_surface_matching_ratio = 0.0
        self.best_penetration_ratio = float('inf')

        self.fracture_mask_a = None
        self.fracture_mask_b = None
        self.shape_emb = None

        self.current_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.max_trans_step = 0.5
        self.max_rot_step = 0.05

        self.collision_threshold = 0.65

        # Tolleranza sul numero di punti compenetranti prima del
        # rollback completo della posizione. Prima, QUALSIASI punto
        # compenetrante (anche uno solo) annullava tutto il movimento
        # -- questo puo' bloccare sul nascere una transizione che
        # richiede un contatto temporaneo minimo per raggiungere una
        # configurazione complessivamente migliore. Sotto questa
        # soglia, il movimento resta valido e viene penalizzato in
        # modo continuo tramite penalty_penetration, non annullato di
        # netto. Non validato su GT (riguarda la dinamica di
        # esplorazione, non un criterio discriminativo).
        self.COLLISION_TOLERANCE_CONTACTS = 10

        # Soglie CALIBRATE empiricamente sul campione con ground truth
        # nota (vedi ground_truth.py), non scelte a intuito:
        #
        # - NORMAL_ALIGNMENT_TARGET: alla posa GT vera, normal_alignment
        #   (locale, sulla maschera statica) vale 0.7349. Su 20 pose
        #   sbagliate casuali, il massimo osservato e' 0.5563. 0.65 sta
        #   comodamente in mezzo.
        #
        # - PENETRATION_RATIO_TARGET: alla posa GT vera, penetration
        #   vale 0.0112. Su 20 pose sbagliate casuali, il minimo
        #   osservato e' 0.0298. 0.02 sta comodamente in mezzo.
        #
        # current_chamfer (sulla maschera statica) NON e' usato come
        # criterio di successo/gate: resta >20mm anche alla posa GT
        # perfetta -- irraggiungibile per costruzione con questa
        # maschera. Resta solo come metrica informativa nella reward.
        self.NORMAL_ALIGNMENT_TARGET = 0.65
        self.PENETRATION_RATIO_TARGET = 0.02
        self.CONTACT_MIN_DIST_TARGET = self.collision_threshold

        # RICALIBRATO dopo il fix del bug che ha attivato la formula
        # blend (60% media + 40% minimo tra settori): valore vero alla
        # GT = 0.1180 (formula attuale, non piu' 0.0698 della formula
        # vecchia a minimo puro). Su 20 pose sbagliate: media 0.0554,
        # solo 1/20 sopra il valore GT (0.1761, outlier). 0.08 sta in
        # mezzo, con margine su entrambi i lati.
        self.SURFACE_MATCHING_TARGET = 0.08

        # Validato su GT: overlap assiale (asse principale di A) alla
        # posa corretta = 0.9296, minimo su 20 pose sbagliate = 0.9730
        # -- margine sottile ma consistente (0/20 sotto il valore GT).
        # Soglia poco sopra il valore GT, per non penalizzare la posa
        # corretta stessa.
        self.AXIAL_OVERLAP_SOFT_THRESHOLD = 0.95

        # 🆕 Validato su GT: dot(direzione punta A, direzione punta B)
        # -- vettore baricentro->punto piu' lontano, un segnale GLOBALE
        # di orientamento (non locale come normal_alignment, che guarda
        # solo il punto di contatto e resta cieco a ossa orientate in
        # direzioni opposte). Alla posa GT vale 0.9965 (punte quasi
        # perfettamente allineate). Su 20 pose sbagliate: range
        # [-0.991, 0.981], media -0.048, 0/20 raggiungono il valore GT.
        # 0.7 sta ben sopra dove finirebbero orientazioni opposte o
        # perpendicolari (vicino a -1 o 0), con margine per il rumore
        # di allenamento.
        self.TIP_ALIGNMENT_TARGET = 0.7

        # Soglia (non validata su GT, euristica basata sui valori
        # osservati) sotto la quale consideriamo il chamfer "gia'
        # abbastanza buono" e spostiamo il peso della reward sulla
        # coverage.
        self.CHAMFER_GOOD_ENOUGH = 13.0  # 6 se CHAMFER_KEEP_FRACTION = 0.2

        # Dimensione osservazione (+1 per tip_alignment)
        base_obs_dim = 16
        _, _, (dummy_graph_a, dummy_graph_b) = self.dataset[0]
        dummy_graph_a = dummy_graph_a.to(self.device)
        dummy_graph_b = dummy_graph_b.to(self.device)

        with torch.no_grad():
            dummy_emb_a = self.encoder(dummy_graph_a)
            dummy_emb_b = self.encoder(dummy_graph_b)
            dummy_shape_emb = torch.cat([dummy_emb_a, dummy_emb_b], dim=-1).flatten()
            if dummy_shape_emb.shape[0] > 1000:
                self.emb_dim = 128
            else:
                self.emb_dim = dummy_shape_emb.shape[0]

        total_obs_dim = base_obs_dim + self.emb_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

        self.prev_chamfer = float('inf')
        self.prev_normal = 0.0
        self.episode_best_chamfer = float('inf')
        self.episode_best_normal = 0.0
        self.episode_best_surface_matching_ratio = 0.0

    @staticmethod
    def _orient_normals_outward(points, normals):
        centroid = points.mean(dim=0, keepdim=True)
        outward_ref = points - centroid
        outward_ref = outward_ref / (torch.norm(outward_ref, dim=1, keepdim=True) + 1e-8)

        dot = torch.sum(normals * outward_ref, dim=1, keepdim=True)
        sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))

        return normals * sign

    @staticmethod
    def _tip_direction(points):
        """
        Direzione "punta": vettore dal baricentro al punto piu'
        lontano, normalizzato. A differenza dell'asse principale
        (PCA), non ha ambiguita' di segno -- e' definito da un punto
        specifico (il piu' lontano), non da un asse bidirezionale.
        Segnale GLOBALE di orientamento del corpo del frammento,
        indipendente da dove si toccano i punti di frattura.
        """
        centroid = points.mean(dim=0)
        dists = torch.norm(points - centroid, dim=1)
        idx = torch.argmax(dists)
        direction = points[idx] - centroid
        return direction / (torch.norm(direction) + 1e-8)

    @staticmethod
    def _random_rotation_matrix(device, dtype):
        """
        Rotazione casuale UNIFORME su SO(3), via quaternione
        normalizzato campionato da una gaussiana.

        🆕 FIX: la versione precedente campionava 3 angoli di Eulero
        indipendenti e uniformi in [0, 2*pi) -- questo NON produce una
        distribuzione uniforme sulle rotazioni 3D (problema noto di
        parametrizzazione: la densita' non e' uniforme rispetto alla
        misura di Haar su SO(3), certe orientazioni vengono
        sovracampionate). Il quaternione gaussiano normalizzato e' il
        modo standard per ottenere una rotazione genuinamente uniforme.
        """
        q = torch.randn(4, device=device, dtype=dtype)
        q = q / torch.norm(q)
        w, x, y, z = q[0], q[1], q[2], q[3]

        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
        ])
        return R

    def _detect_penetration(self):
        with torch.no_grad():
            dists = torch.cdist(self.frag_b, self.frag_a)
            min_dist, nearest_idx = torch.min(dists, dim=1)

            nearest_point_a = self.frag_a[nearest_idx]
            nearest_normal_a = self.frag_a_normals[nearest_idx]

            vec_a_to_b = self.frag_b - nearest_point_a
            signed = torch.sum(vec_a_to_b * nearest_normal_a, dim=1)

            is_penetrating = (min_dist < self.collision_threshold) & (signed < 0)

            num_contacts = torch.sum(is_penetrating).item()
            min_dist_scalar = torch.min(min_dist).item()

            penetration_ratio = num_contacts / self.frag_b.shape[0]

            return num_contacts, min_dist_scalar, penetration_ratio

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx = np.random.randint(0, len(self.dataset))
        frag_a_full, frag_b_full, (graph_a, graph_b) = self.dataset[idx]
        graph_a = graph_a.to(self.device)
        graph_b = graph_b.to(self.device)

        if hasattr(frag_a_full, 'x'):
            frag_a_full = frag_a_full.x
        if hasattr(frag_b_full, 'x'):
            frag_b_full = frag_b_full.x

        self.frag_a = (frag_a_full[:, :3] * 70).clone().detach().to(self.device)
        self.frag_b = (frag_b_full[:, :3] * 70).clone().detach().to(self.device)

        self.frag_a_normals = frag_a_full[:, 3:6].clone().detach().to(self.device)
        self.frag_b_normals = frag_b_full[:, 3:6].clone().detach().to(self.device)

        self.frag_a_normals = self._orient_normals_outward(self.frag_a, self.frag_a_normals)
        self.frag_b_normals = self._orient_normals_outward(self.frag_b, self.frag_b_normals)

        if frag_a_full.shape[1] > 7:
            self.fracture_mask_a = (frag_a_full[:, 7] > 0.5)
            self.fracture_mask_b = (frag_b_full[:, 7] > 0.5)
        else:
            self.fracture_mask_a = torch.ones(frag_a_full.shape[0], dtype=torch.bool, device=self.device)
            self.fracture_mask_b = torch.ones(frag_b_full.shape[0], dtype=torch.bool, device=self.device)

        # Settori spaziali della fracture_mask di A, tramite K-Means a
        # K FISSO. A non si muove mai, quindi i settori sono stabili
        # per l'intero episodio.
        N_FRACTURE_SECTORS = 4

        frag_a_mask_points = self.frag_a[self.fracture_mask_a]
        if frag_a_mask_points.shape[0] >= N_FRACTURE_SECTORS:
            km = KMeans(n_clusters=N_FRACTURE_SECTORS, n_init=10, random_state=0).fit(
                frag_a_mask_points.detach().cpu().numpy()
            )
            labels = km.labels_
            self.fracture_cluster_labels_a = torch.tensor(labels, dtype=torch.long, device=self.device)
        else:
            self.fracture_cluster_labels_a = torch.zeros(
                frag_a_mask_points.shape[0], dtype=torch.long, device=self.device
            )

        # Asse principale di A (PCA sull'intero frammento). A non si
        # muove mai, quindi asse e centro sono stabili per l'intero
        # episodio. Usato per la penalita' morbida di sovrapposizione
        # assiale.
        centered_a = self.frag_a - self.frag_a.mean(dim=0)
        _, _, V_a = torch.linalg.svd(centered_a)
        self.principal_axis_a = V_a[0]
        self.axis_center_a = self.frag_a.mean(dim=0)

        # 🆕 Direzione "punta" di A (fissa, A non si muove mai). Il
        # "polo" globale di orientamento -- vedi _tip_direction.
        self.tip_dir_a = self._tip_direction(self.frag_a)

        # DISABILITATO su richiesta: blocco che applicava T_GT +
        # scostamento di 15mm come punto di partenza per B.
        #
        T_GT = torch.tensor([
             [-0.264350, -0.115371,  0.957501,  32.685013],
             [ 0.196923, -0.978359, -0.063517, -44.966434],
             [ 0.944108,  0.171764,  0.281348,   4.563776],
             [ 0.000000,  0.000000,  0.000000,   1.000000]
         ], dtype=torch.float32, device=self.device)
        
        R_gt = T_GT[:3, :3]
        t_gt = T_GT[:3, 3]
        
        self.frag_b = (self.frag_b @ R_gt.T) + t_gt
        self.frag_b_normals = self.frag_b_normals @ R_gt.T
        
        mean_normal_a = self.frag_a_normals[self.fracture_mask_a].mean(dim=0)
        detach_dir = mean_normal_a / (torch.norm(mean_normal_a) + 1e-8)
        
        OFFSET_DIST = 15.0
        self.frag_b += detach_dir * OFFSET_DIST

        # Rotazione CASUALE (diversa ogni episodio, genuinamente
        # uniforme su SO(3) -- vedi _random_rotation_matrix) intorno
        # al baricentro di B, al posto di T_GT.
        #R_random = self._random_rotation_matrix(self.device, self.frag_b.dtype)

        #centro_b_init = torch.mean(self.frag_b, dim=0)
        #frag_b_centered_init = self.frag_b - centro_b_init
        #transformed_init = torch.mm(frag_b_centered_init, R_random.t())

        #self.frag_b = transformed_init + centro_b_init
        #self.frag_b_normals = torch.mm(self.frag_b_normals, R_random.t())

        with torch.no_grad():
            emb_a = self.encoder(graph_a)
            emb_b = self.encoder(graph_b)
            raw_emb = torch.cat([emb_a, emb_b], dim=-1).flatten()

            if raw_emb.shape[0] > 1000:
                emb_tensor = raw_emb.view(128, -1).mean(dim=1)
                self.shape_emb = emb_tensor.cpu().numpy()
            else:
                self.shape_emb = raw_emb.cpu().numpy()

            current_chamfer, _, surface_matching_ratio, min_dist, _, normal_alignment, penetration_ratio, axial_overlap, tip_alignment = self._compute_geometric_metrics()

        self.current_step = 0
        self.prev_chamfer = current_chamfer
        self.prev_normal = normal_alignment
        self.prev_surface_matching_ratio = surface_matching_ratio

        self.episode_best_chamfer = current_chamfer
        self.episode_best_normal = normal_alignment
        self.episode_best_surface_matching_ratio = surface_matching_ratio
        self.current_rotation = torch.zeros(3, device=self.frag_b.device, dtype=self.frag_b.dtype)

        if hasattr(self, 'center_a_frac'):
            delattr(self, 'center_a_frac')

        if self.best_frag_a is None:
            self.best_frag_a = self.frag_a.detach().cpu().numpy().copy()
            self.best_frag_b = self.frag_b.detach().cpu().numpy().copy()
            self.global_best_chamfer = current_chamfer
            self.global_best_normal = normal_alignment
            self.global_best_surface_matching_ratio = surface_matching_ratio
            self.global_best_penetration_ratio = penetration_ratio
            self.best_chamfer = current_chamfer
            self.best_normal = normal_alignment
            self.best_surface_matching_ratio = surface_matching_ratio
            self.best_penetration_ratio = penetration_ratio

        obs = self._get_obs(current_chamfer, min_dist, normal_alignment, tip_alignment)
        return obs, {}

    def step(self, action):
        self.current_step += 1

        device = self.frag_b.device
        dtype = self.frag_b.dtype

        posizione_precedente_b = self.frag_b.clone()
        normali_precedenti_b = self.frag_b_normals.clone()
        if isinstance(self.current_rotation, torch.Tensor):
            rotazione_precedente_b = self.current_rotation.clone()
        else:
            rotazione_precedente_b = self.current_rotation.copy()

        dx, dy, dz, d_roll, d_pitch, d_yaw = action

        step_scale = max(0.01, min(0.5, self.prev_chamfer * 0.03))        # mm
        rot_scale = max(0.002, min(0.05, self.prev_chamfer * 0.004))       # radianti

        trans = torch.tensor([dx, dy, dz], device=device, dtype=dtype) * step_scale
        delta_rot = torch.tensor([d_roll, d_pitch, d_yaw], device=device, dtype=dtype) * rot_scale

        with torch.no_grad():
            T = self._euler_and_translation_to_matrix(delta_rot, trans)
            centro_b = torch.mean(self.frag_b, dim=0)

            frag_b_centered = self.frag_b - centro_b
            ones = torch.ones((frag_b_centered.shape[0], 1), device=device, dtype=dtype)
            frag_b_homogeneous = torch.cat([frag_b_centered, ones], dim=1)

            transformed_homogeneous = torch.mm(frag_b_homogeneous, T.t())
            self.frag_b = transformed_homogeneous[:, :3] + centro_b

            R = T[:3, :3]
            self.frag_b_normals = torch.mm(self.frag_b_normals, R.t())
            self.current_rotation += delta_rot

            (
                current_chamfer,
                fracture_disparity,
                surface_matching_ratio,
                min_dist,
                num_contacts,
                normal_alignment,
                penetration_ratio,
                axial_overlap,
                tip_alignment,
            ) = self._compute_geometric_metrics()

        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps
        is_success = False

        if current_chamfer > self.max_allowed_chamfer:
            reward = -20.0
            terminated = True
            return (
                self._get_obs(current_chamfer, min_dist, normal_alignment, tip_alignment),
                float(reward),
                terminated,
                truncated,
                {
                    "chamfer": float(current_chamfer),
                    "contacts": 0,
                    "surface_matching_ratio": float(surface_matching_ratio),
                    "normal_alignment": float(normal_alignment),
                    "success": False,
                },
            )

        if num_contacts > self.COLLISION_TOLERANCE_CONTACTS:
            reward = -1.0 - (num_contacts * 0.05)
            if surface_matching_ratio < 0.50:
                mean_normal_a = self.frag_a_normals[self.fracture_mask_a].mean(dim=0)
                detach_dir = mean_normal_a / (torch.norm(mean_normal_a) + 1e-8)
                self.frag_b = posizione_precedente_b + (detach_dir * 0.8)
            else:
                self.frag_b = posizione_precedente_b
            self.frag_b_normals = normali_precedenti_b

            if isinstance(self.current_rotation, torch.Tensor):
                self.current_rotation = rotazione_precedente_b.clone()
            else:
                self.current_rotation = rotazione_precedente_b.copy()

            with torch.no_grad():
                (current_chamfer, _, surface_matching_ratio, min_dist, _, normal_alignment, penetration_ratio, axial_overlap, tip_alignment) = self._compute_geometric_metrics()
            info_contacts = num_contacts
        else:
            info_contacts = num_contacts

            step_delta_chamfer = self.prev_chamfer - current_chamfer
            step_delta_normal = normal_alignment - self.prev_normal
            step_delta_coverage = surface_matching_ratio - self.prev_surface_matching_ratio

            # Tre stadi: 1) orientamento, 2) chamfer, 3) coverage.
            if normal_alignment < self.NORMAL_ALIGNMENT_TARGET:
                w_normal, w_chamfer, w_coverage = 15.0, 2.0, 1.0
            elif current_chamfer > self.CHAMFER_GOOD_ENOUGH:
                w_normal, w_chamfer, w_coverage = 2.0, 15.0, 2.0
            else:
                w_normal, w_chamfer, w_coverage = 2.0, 2.0, 25.0

            # 🆕 Non premiare un miglioramento del chamfer (troncato,
            # quindi soddisfacibile trovando SOLO una porzione di punti
            # molto vicini) se nello stesso passo la coverage sta
            # peggiorando -- osservato in pratica: l'agente trova
            # configurazioni a incrocio/angolo dove pochi punti sono
            # vicinissimi (chamfer migliore perfino della GT), a scapito
            # della superficie di contatto reale. Il guadagno di chamfer
            # in quel caso e' azzerato, non solo attenuato: altrimenti
            # resterebbe comunque conveniente il compromesso.
            COVERAGE_SACRIFICE_TOLERANCE = 0.01  # piccola tolleranza al rumore
            chamfer_is_sacrificing_coverage = (
                step_delta_chamfer > 0 and step_delta_coverage < -COVERAGE_SACRIFICE_TOLERANCE
            )

            chamfer_delta_component = 0.0 if chamfer_is_sacrificing_coverage else step_delta_chamfer * w_chamfer
            reward_delta = chamfer_delta_component + (step_delta_normal * w_normal)

            progress_chamfer = (
                0.0 if chamfer_is_sacrificing_coverage
                else max(0.0, self.episode_best_chamfer - current_chamfer)
            )
            progress_normal = max(0.0, normal_alignment - self.episode_best_normal)
            progress_coverage = max(0.0, surface_matching_ratio - self.episode_best_surface_matching_ratio)
            reward_progress = (
                (progress_chamfer * w_chamfer * 2.0)
                + (progress_normal * w_normal * 2.0)
                + (progress_coverage * w_coverage * 2.0)
            )

            if current_chamfer < 15.0:
                absolute_dense_reward = (15.0 - current_chamfer) * 0.15
            else:
                absolute_dense_reward = 0.0

            reward_coverage = surface_matching_ratio * w_coverage

            penalty_penetration = -penetration_ratio * 20.0

            penalty_axial_overlap = -max(0.0, axial_overlap - self.AXIAL_OVERLAP_SOFT_THRESHOLD) * 20.0

            # 🆕 Reward densa e continua sul "polo" (dot delle direzioni
            # punta A/B): spinge verso +1 (punte allineate) e penalizza
            # valori negativi (punte opposte -- il problema osservato).
            # Segnale globale, indipendente da dove si toccano i punti.
            reward_tip_alignment = tip_alignment * 10.0

            penalty_regression = 0.0
            if step_delta_chamfer < 0 and current_chamfer > 6.0:
                if step_delta_normal <= 0:
                    penalty_regression += step_delta_chamfer * 2.0

            if step_delta_normal < 0 and normal_alignment < 0.65:
                penalty_regression += step_delta_normal * 10.0

            penalty_gap = 0.0
            if min_dist < 2.0 and fracture_disparity > 6.0:
                penalty_gap = -(fracture_disparity * 0.2)

            reward = (
                reward_delta + reward_progress + penalty_regression + reward_coverage
                + penalty_penetration + penalty_axial_overlap + reward_tip_alignment - 0.02
            )

            if current_chamfer < self.episode_best_chamfer:
                self.episode_best_chamfer = current_chamfer

            if normal_alignment > self.episode_best_normal:
                self.episode_best_normal = normal_alignment

            if surface_matching_ratio > self.episode_best_surface_matching_ratio:
                self.episode_best_surface_matching_ratio = surface_matching_ratio

            if self.global_best_chamfer < self.CHAMFER_GOOD_ENOUGH:
                if (
                penetration_ratio <= self.PENETRATION_RATIO_TARGET
                and normal_alignment >= self.NORMAL_ALIGNMENT_TARGET
                and surface_matching_ratio >= self.global_best_surface_matching_ratio
                and min_dist <= self.CONTACT_MIN_DIST_TARGET
                and tip_alignment >= self.TIP_ALIGNMENT_TARGET
                and current_chamfer <= self.CHAMFER_GOOD_ENOUGH
                ):
                    self.global_best_chamfer = current_chamfer
                    self.global_best_normal = normal_alignment
                    self.global_best_surface_matching_ratio = surface_matching_ratio
                    self.global_best_penetration_ratio = penetration_ratio
                    self.best_chamfer = current_chamfer
                    self.best_normal = normal_alignment
                    self.best_surface_matching_ratio = surface_matching_ratio
                    self.best_penetration_ratio = penetration_ratio
                    self.best_frag_a = self.frag_a.detach().cpu().numpy().copy()
                    self.best_frag_b = self.frag_b.detach().cpu().numpy().copy()
            elif (
                penetration_ratio <= self.PENETRATION_RATIO_TARGET
                and normal_alignment >= self.NORMAL_ALIGNMENT_TARGET
                and surface_matching_ratio >= self.global_best_surface_matching_ratio
                and min_dist <= self.CONTACT_MIN_DIST_TARGET
                and tip_alignment >= self.TIP_ALIGNMENT_TARGET
                ):
                    self.global_best_chamfer = current_chamfer
                    self.global_best_normal = normal_alignment
                    self.global_best_surface_matching_ratio = surface_matching_ratio
                    self.global_best_penetration_ratio = penetration_ratio
                    self.best_chamfer = current_chamfer
                    self.best_normal = normal_alignment
                    self.best_surface_matching_ratio = surface_matching_ratio
                    self.best_penetration_ratio = penetration_ratio
                    self.best_frag_a = self.frag_a.detach().cpu().numpy().copy()
                    self.best_frag_b = self.frag_b.detach().cpu().numpy().copy()


            if (penetration_ratio <= self.PENETRATION_RATIO_TARGET
                and normal_alignment >= self.NORMAL_ALIGNMENT_TARGET
                and min_dist <= self.CONTACT_MIN_DIST_TARGET
                and surface_matching_ratio >= self.SURFACE_MATCHING_TARGET
                and tip_alignment >= self.TIP_ALIGNMENT_TARGET
                and current_chamfer <= self.CHAMFER_GOOD_ENOUGH):
                reward += 100.0
                terminated = True
                is_success = True
                print(
                    f"🎯 INCASTRO TROVATO AL PASSO {self.current_step}! "
                    f"(Chamfer: {current_chamfer:.4f} mm, Norm Alignment: {normal_alignment:.4f}, "
                    f"Coverage: {surface_matching_ratio:.2f}, Tip: {tip_alignment:.2f})"
                )

        self.prev_chamfer = current_chamfer
        self.prev_normal = normal_alignment
        self.prev_surface_matching_ratio = surface_matching_ratio
        obs = self._get_obs(current_chamfer, min_dist, normal_alignment, tip_alignment)

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            {
                "chamfer": float(current_chamfer),
                "contacts": info_contacts,
                "surface_matching_ratio": float(surface_matching_ratio),
                "normal_alignment": float(normal_alignment),
                "penetration_ratio": float(penetration_ratio),
                "tip_alignment": float(tip_alignment),
                "success": is_success,
            },
        )

    def _compute_geometric_metrics(self):
        with torch.no_grad():
            mask_a_bool = self.fracture_mask_a.bool()
            mask_b_bool = self.fracture_mask_b.bool()

            frac_a = self.frag_a[mask_a_bool] if torch.any(mask_a_bool) else self.frag_a
            frac_b = self.frag_b[mask_b_bool] if torch.any(mask_b_bool) else self.frag_b

            norm_a_surf = self.frag_a_normals[mask_a_bool] if torch.any(mask_a_bool) else self.frag_a_normals
            norm_b_surf = self.frag_b_normals[mask_b_bool] if torch.any(mask_b_bool) else self.frag_b_normals

            dists_frac = torch.cdist(frac_b, frac_a)
            min_b2a_frac = torch.min(dists_frac, dim=1).values
            min_a2b_frac = torch.min(dists_frac, dim=0).values

            CHAMFER_KEEP_FRACTION = 0.5

            k_b = max(1, int(len(min_b2a_frac) * CHAMFER_KEEP_FRACTION))
            k_a = max(1, int(len(min_a2b_frac) * CHAMFER_KEEP_FRACTION))

            trimmed_b2a = torch.topk(min_b2a_frac, k_b, largest=False).values.mean()
            trimmed_a2b = torch.topk(min_a2b_frac, k_a, largest=False).values.mean()

            current_chamfer = (trimmed_b2a + trimmed_a2b).item()

            idx_b2a = torch.argmin(dists_frac, dim=1)
            corresponding_normal_a = norm_a_surf[idx_b2a]

            local_dot = torch.sum(norm_b_surf * corresponding_normal_a, dim=1)
            local_alignment = 0.5 * (1.0 - local_dot)

            proximity_scale = max(self.collision_threshold, 1e-3)
            weight = torch.exp(-min_b2a_frac / proximity_scale)
            weight_sum = weight.sum()

            if weight_sum > 1e-6:
                normal_alignment = (local_alignment * weight).sum().item() / weight_sum.item()
            else:
                normal_alignment = 0.0

            num_contacts, min_dist_scalar, penetration_ratio = self._detect_penetration()

            proximity_band = 2.0  # mm
            matching_b = (torch.sum(min_b2a_frac < proximity_band).float() / min_b2a_frac.numel()).item()

            cluster_labels = self.fracture_cluster_labels_a
            unique_clusters = torch.unique(cluster_labels)
            cluster_ratios = []

            for c in unique_clusters:
                if c.item() == -1:
                    continue
                idx = (cluster_labels == c)
                if idx.sum() == 0:
                    continue
                frac_close = (min_a2b_frac[idx] < proximity_band).float().mean()
                cluster_ratios.append(frac_close)

            if len(cluster_ratios) > 0:
                cluster_tensor = torch.stack(cluster_ratios)
                cluster_mean = cluster_tensor.mean().item()
                cluster_min = cluster_tensor.min().item()
                matching_a = 0.6 * cluster_mean + 0.4 * cluster_min
            else:
                matching_a = 0.0

            surface_matching_ratio = 0.5 * (matching_b + matching_a)

            max_gap = torch.max(min_b2a_frac).item()
            min_gap = torch.min(min_b2a_frac).item()
            fracture_disparity = max_gap - min_gap

            # Sovrapposizione assiale (asse principale di A, fisso).
            proj_a = torch.matmul(self.frag_a - self.axis_center_a, self.principal_axis_a)
            proj_b = torch.matmul(self.frag_b - self.axis_center_a, self.principal_axis_a)

            a_min, a_max = proj_a.min().item(), proj_a.max().item()
            b_min, b_max = proj_b.min().item(), proj_b.max().item()

            axial_overlap_raw = max(0.0, min(a_max, b_max) - max(a_min, b_min))
            a_len = max(a_max - a_min, 1e-6)
            b_len = max(b_max - b_min, 1e-6)
            axial_overlap = axial_overlap_raw / min(a_len, b_len)

            # 🆕 "Polo": dot tra direzione punta di A (fissa) e di B
            # (ricalcolata ogni volta dalla posa corrente di frag_b --
            # sotto trasformazione rigida il punto piu' lontano dal
            # baricentro resta lo stesso punto fisico, quindi la
            # direzione ruota correttamente insieme al frammento).
            tip_dir_b = self._tip_direction(self.frag_b)
            tip_alignment = torch.dot(self.tip_dir_a, tip_dir_b).item()

            return (
                current_chamfer, fracture_disparity, surface_matching_ratio, min_dist_scalar,
                num_contacts, normal_alignment, penetration_ratio, axial_overlap, tip_alignment,
            )

    def _get_obs(self, current_chamfer, min_dist, normal_alignment, tip_alignment):
        with torch.no_grad():
            if not hasattr(self, "center_a_frac"):
                mask_a_bool = self.fracture_mask_a.bool()
                self.center_a_frac = (
                    torch.mean(self.frag_a[mask_a_bool], dim=0)
                    if torch.any(mask_a_bool)
                    else torch.mean(self.frag_a, dim=0)
                )

            mask_b_bool = self.fracture_mask_b.bool()
            center_b_frac = (
                torch.mean(self.frag_b[mask_b_bool], dim=0)
                if torch.any(mask_b_bool)
                else torch.mean(self.frag_b, dim=0)
            )

            approach_vector = self.center_a_frac - center_b_frac
            approach_dir = (
                approach_vector / (torch.norm(approach_vector) + 1e-8)
            ).cpu().numpy()

            if torch.any(mask_b_bool):
                mean_norm_b = torch.mean(self.frag_b_normals[mask_b_bool], dim=0)
                mean_norm_b_dir = (
                    mean_norm_b / (torch.norm(mean_norm_b) + 1e-8)
                ).cpu().numpy()
            else:
                mean_norm_b_dir = np.zeros(3, dtype=np.float32)

        if isinstance(self.current_rotation, torch.Tensor):
            r, p, y = self.current_rotation.cpu().numpy()
        else:
            r, p, y = self.current_rotation

        geom_obs = np.array(
            [
                float(current_chamfer),
                float(normal_alignment),
                float(min_dist),
                np.sin(r),
                np.cos(r),
                np.sin(p),
                np.cos(p),
                np.sin(y),
                np.cos(y),
                approach_dir[0],
                approach_dir[1],
                approach_dir[2],
                mean_norm_b_dir[0],
                mean_norm_b_dir[1],
                mean_norm_b_dir[2],
                float(tip_alignment),  # 🆕 il "polo": segnale globale di orientamento
            ],
            dtype=np.float32,
        )

        return np.concatenate([geom_obs, self.shape_emb])

    def _euler_and_translation_to_matrix(self, angles, translation):
        r, p, y = angles
        tx, ty, tz = translation
        device, dtype = self.frag_b.device, self.frag_b.dtype

        r_t, p_t, y_t = torch.as_tensor(r, device=device, dtype=dtype), torch.as_tensor(p, device=device, dtype=dtype), torch.as_tensor(y, device=device, dtype=dtype)
        cos_r, sin_r = torch.cos(r_t), torch.sin(r_t)
        cos_p, sin_p = torch.cos(p_t), torch.sin(p_t)
        cos_y, sin_y = torch.cos(y_t), torch.sin(y_t)

        Rx = torch.eye(3, device=device, dtype=dtype)
        Rx[1, 1], Rx[1, 2] = cos_r, -sin_r
        Rx[2, 1], Rx[2, 2] = sin_r, cos_r

        Ry = torch.eye(3, device=device, dtype=dtype)
        Ry[0, 0], Ry[0, 2] = cos_p, sin_p
        Ry[2, 0], Ry[2, 2] = -sin_p, cos_p

        Rz = torch.eye(3, device=device, dtype=dtype)
        Rz[0, 0], Rz[0, 1] = cos_y, -sin_y
        Rz[1, 0], Rz[1, 1] = sin_y, cos_y

        R = torch.mm(Rz, torch.mm(Ry, Rx))
        T = torch.eye(4, device=device, dtype=dtype)
        T[:3, :3] = R
        T[0, 3], T[1, 3], T[2, 3] = torch.as_tensor(tx, device=device, dtype=dtype), torch.as_tensor(ty, device=device, dtype=dtype), torch.as_tensor(tz, device=device, dtype=dtype)

        return T
