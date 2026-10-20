"""
Scelta adattiva del roll di partenza (start_mode="tip_aligned").

Con un roll continuo su 360 gradi quasi tutte le partenze hanno un roll
sbagliato, da cui avvicinarsi produce solo contatti poveri: l'agente non
impara mai l'avvicinamento. Qui le partenze usano pochi roll fissi (come
la vecchia partenza T_GT era una posa fissa) e gli episodi vengono
concentrati sui roll che producono il potenziale Phi migliore.

Nessuna ground truth: il punteggio di un roll e' la media mobile del Phi
massimo (senza compenetrazione) raggiunto negli episodi partiti da quel
roll, cioe' la stessa quantita' che il reward premia.
"""
import numpy as np


class RollSampler:
    def __init__(self, rolls_deg, uniform_prob=0.5, temperature=2.0, ema=0.1):
        self.rolls = [float(r) for r in rolls_deg]
        self.uniform_prob = uniform_prob   # quota di episodi sempre uniforme
        self.temperature = temperature
        self.ema = ema
        self.scores = np.full(len(self.rolls), np.nan)
        self.counts = np.zeros(len(self.rolls), dtype=np.int64)

    def probabilities(self):
        n = len(self.rolls)
        if np.all(np.isnan(self.scores)):
            return np.full(n, 1.0 / n)
        # roll mai provati: punteggio pari al migliore, per provarli
        scores = np.where(np.isnan(self.scores), np.nanmax(self.scores), self.scores)
        z = (scores - scores.max()) / self.temperature
        soft = np.exp(z) / np.exp(z).sum()
        return self.uniform_prob / n + (1 - self.uniform_prob) * soft

    def sample(self, rng):
        i = int(rng.choice(len(self.rolls), p=self.probabilities()))
        return i, self.rolls[i]

    def update(self, index, best_phi):
        if best_phi is None:
            return
        if np.isnan(self.scores[index]):
            self.scores[index] = best_phi
        else:
            self.scores[index] = (1 - self.ema) * self.scores[index] + self.ema * best_phi
        self.counts[index] += 1

    def best(self):
        if np.all(np.isnan(self.scores)):
            return None, None
        i = int(np.nanargmax(self.scores))
        return self.rolls[i], float(self.scores[i])

    def state(self):
        return {"roll_sampler_scores": self.scores.copy(), "roll_sampler_counts": self.counts.copy(),
                "roll_sampler_rolls": np.array(self.rolls)}

    def load_state(self, records):
        if "roll_sampler_scores" not in records:
            return
        if list(np.asarray(records["roll_sampler_rolls"], dtype=float)) != self.rolls:
            print("⚠️ Roll del checkpoint diversi da quelli attuali: punteggi dei roll ripartono da zero.")
            return
        self.scores = np.asarray(records["roll_sampler_scores"], dtype=float).copy()
        self.counts = np.asarray(records["roll_sampler_counts"], dtype=np.int64).copy()
