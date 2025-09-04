# This file is used to run a series of experiments in sequence.

import subprocess
import os


log_path = "experiment_logs"
script = "python .\\scripts\\population_evolution.py" #"python .\\scripts\\sb3\\train.py"
parameters = ["--task Grasp-Objects --num_envs 1024 --headless --generations 10"]*5 #["--task Grasp-Pot --num_envs 1024 --headless --max_steps 2600000 --eval_frequency 260000"] * 5 
starting_seed = 200


if __name__ == "__main__":
    os.makedirs(log_path, exist_ok=True)
    for ind, params in enumerate(parameters):
        print(f"Running experiment: {ind}")
        exp = f"{script} {params}"
        seed = starting_seed + ind
        exp += f" --seed {seed}"
        print(f"\tCommand: {exp}")
        log_file = os.path.join(log_path, f"experiment_{ind}.log")
        with open(log_file, "w") as f:
            process = subprocess.Popen(exp, shell=True, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
        print(f"Experiment finished: {ind}")