# meta_controller.py — Oversees RealignR training phases

import time

class MetaController:
    def __init__(self, plateau_window=400, tol=1e-4):
        self.loss_hist = []
        self.action_hist = []
        self.plateau_window = plateau_window
        self.tol = tol
        self.cpr_timer = -1  # ‑1  →  CPR inactive
        self.cpr_active = False  # For compatibility with training loop

    def update(self, step, loss, g_mean, cpr_trigger=0):
        self.loss_hist.append(loss)
        if len(self.loss_hist) > self.plateau_window * 3:
            self.loss_hist = self.loss_hist[-self.plateau_window * 3:]
        if self._on_plateau(self.loss_hist):
            self.trigger_CPR()

    def _on_plateau(self, hist):
        if len(hist) < self.plateau_window:
            return False
        slope = (hist[-1] - hist[-self.plateau_window]) / self.plateau_window
        return abs(slope) < self.tol

    def trigger_CPR(self):
        if self.cpr_timer > 0:
            return
        print(f"⚠️  CPR Triggered at {time.strftime('%H:%M:%S')}")
        self.cpr_timer = 600  # e.g. 600 steps

class CPRController:
    def __init__(self, epsilon=1e-3, reset_patience=500):
        self.epsilon = epsilon
        self.reset_patience = reset_patience
        self.loss_history = []
        self.trigger_count = 0

    def update(self, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) > self.reset_patience:
            self.loss_history.pop(0)

        if len(self.loss_history) >= self.reset_patience:
            avg_loss = sum(self.loss_history) / len(self.loss_history)
            if abs(loss - avg_loss) > self.epsilon:
                self.trigger_count += 1
                return "TRIGGERED"
            else:
                return "RESET"
        return "STABLE"
