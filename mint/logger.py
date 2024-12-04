import os.path
import shutil

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LoggerConfig:
    log_dir: str = "../runs/"
    experiment_name: str = "default_experiment"
    log_frequency: int = 10


class Logger:
    def __init__(self, log_dir, experiment_name, log_frequency):
        # get abs path
        self.log_dir = os.path.join(os.path.abspath(log_dir), experiment_name)
        if os.path.exists(log_dir):
            self.log_dir = f"{self.log_dir}_{datetime.now().strftime('%d_%mT%H_%M')}"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.log_frequency = log_frequency
        self.tensorboard_writer = SummaryWriter(self.log_dir)

        self.metrics = {}
        self.metrics_steps = defaultdict(int)

    def log_scalar(self, tag, value):
        if tag in self.metrics:
            self.metrics[tag].append(value)
            self.metrics_steps[tag] += 1
        else:
            self.metrics[tag] = [value]
            self.metrics_steps[tag] = 1

        if self.metrics_steps[tag] % self.log_frequency == 0:
            self.tensorboard_writer.add_scalar(tag, sum(self.metrics[tag]) / len(self.metrics[tag]), self.metrics_steps[tag])
            self.metrics[tag] = []

    def log_text(self, tag, text):
        if tag in self.metrics_steps:
            self.metrics_steps[tag] += 1
        else:
            self.metrics_steps[tag] = 1

        if self.metrics_steps[tag] % self.log_frequency == 0:
            self.tensorboard_writer.add_text(tag, text, self.metrics_steps[tag])
