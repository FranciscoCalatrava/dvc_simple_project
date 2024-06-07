import subprocess
import random

# Automated random search experiments
num_exps = 10
random.seed(0)

for i in range(1, 9):  # Loop over your 8 experiments
    params = {
        "lr": random.uniform(1e-5, 1e-3),
        "batch_size": random.choice([8, 16, 32, 64, 128, 256])
    }
    exp_name = f"Experiment{i}"
    subprocess.run([
        "dvc", "exp", "run", "--queue",
        "--set-param", f"train.lr={params['lr']}",
        "--set-param", f"train.batch_size={params['batch_size']}",
        "--set-param", f"train.output_dir={i}"
    ])
