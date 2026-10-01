import datetime
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """训练日志记录（支持 TensorBoard 和本地文件）"""

    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir)
        self.log_file = open(os.path.join(log_dir, "training.log"), "w", encoding="utf-8")

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.log_file.write(f"[Step {step}] {tag}: {value:.4f}\n")
        self.log_file.flush()

    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            if isinstance(value, (list, np.ndarray)):
                continue
            self.log_scalar(f"metrics/{key}", value, step)

    def close(self):
        self.writer.close()
        self.log_file.close()

