import torch
import psutil
import os
import time
import threading
import numpy as np
import pandas as pd
from pathlib import Path

class TrainingProfiler:

    def __init__(self, interval=1.0):
        self.interval = interval
        self.process = psutil.Process(os.getpid())

        self.ram_usage = []
        self.vram_usage = []

        self._running = False
        self.thread = None

    def _sample_memory_usage(self):
        while self._running:
            ram = self.process.memory_info().rss / 1024**3
            self.ram_usage.append(ram)

            if torch.cuda.is_available():
                vram = torch.cuda.memory_allocated() / 1024**3
            else:
                vram = 0

            self.vram_usage.append(vram)

            time.sleep(self.interval)

    def start(self):
        self._running = True
        self.start_time = time.time()

        self.thread = threading.Thread(target=self._sample_memory_usage)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self._running = False
        self.thread.join()

        self.end_time = time.time()

    def summary(self):

        return {
            "total_train_time_sec": self.end_time - self.start_time,
            "avg_ram_gb": np.mean(self.ram_usage),
            "avg_vram_gb": np.mean(self.vram_usage),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3
        }
    

def profile_training(
    train_fn,
    model_name,
    num_epochs,
    dataset_size,
    adata_path,
    *args,
    csv_path="training_stats.csv",
    **kwargs
):

    profiler = TrainingProfiler(interval=1)

    torch.cuda.reset_peak_memory_stats()

    profiler.start()

    train_fn(*args, **kwargs)

    profiler.stop()

    results = {}
    results["model_name"] = model_name
    results["num_epochs"] = num_epochs
    results["dataset_size"] = dataset_size
    results["adata_path"] = adata_path
    results.update(profiler.summary())

    df = pd.DataFrame([results])

    if Path(csv_path).exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    return results