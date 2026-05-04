import json
import numpy as np
from collections import defaultdict


class PerformanceEvaluator:
    def __init__(self):
        self.fps = []
        self.lanes = []
        self.obs = []
        self.decisions = defaultdict(int)

    def log_frame(self, fps, left, right, obs, decision):
        self.fps.append(fps)
        self.lanes.append(int(left or right))
        self.obs.append(obs)
        self.decisions[decision] += 1

    def save_report(self, path="metrics_report.json"):
        report = {
            "Average FPS": float(np.mean(self.fps)),
            "Lane Detection Rate": float(np.mean(self.lanes) * 100),
            "Average Obstacles": float(np.mean(self.obs)),
            "Decision Distribution": dict(self.decisions)
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=4)

        print("Metrics saved.")