import optuna
import subprocess
import yaml
import re

def train_agent(trial, learning_rate, gamma, batch_size, hidden_layer_size, buffer_size, entropy_coef):
    config = {
        "behaviors": {
            "MoveToGoal": {
                "trainer_type": "sac",
                "hyperparameters": {
                    "batch_size": batch_size,
                    "buffer_size": buffer_size,
                    "learning_rate": learning_rate,
                    "learning_rate_schedule": "constant",
                    "buffer_init_steps": 10000,
                    "tau": 0.005,
                    "steps_per_update": 5,
                    "save_replay_buffer": False,
                    "init_entcoef": entropy_coef,
                },
                "network_settings": {
                    "hidden_units": hidden_layer_size,
                    "num_layers": 2,
                    "normalize": False
                },
                "reward_signals": {
                    "extrinsic": {
                        "gamma": gamma,
                        "strength": 1.0
                    }
                },
                "max_steps": 500000, 
                "time_horizon": 64,
                "summary_freq": 500
            }
        }
    }
    
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # executing MLAgents command
    result = subprocess.run([
        "mlagents-learn", "config.yaml", "--run-id=optuna_run",
        "--env=../ThesisProject/Builds/ThesisProject.exe",  
        "--force", 
        "--no-graphics", 
        "--width=1920", "--height=1080"
    ], capture_output=True, text=True)
    
    reward = None 
    rewards = [] 
    step = 0 
    
    # Scan output for rewards
    for line in result.stdout.split("\n"):
        if "Mean Reward:" in line:
            match = re.search(r"Mean Reward:\s([-+]?\d*\.\d+|\d+)", line)
            if match:
                try:
                    reward = float(match.group(1))
                    rewards.append(reward)

                    trial.report(reward, step)
                    step += 1

                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                except ValueError:
                    pass
    
    return sum(rewards) / len(rewards) if rewards else -float("inf")

def optimize_hyperparameters(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    entropy_coef = trial.suggest_float('entropy_coef', 0.01, 0.2)
    batch_size = trial.suggest_int('batch_size', 64, 1024)
    hidden_layer_size = trial.suggest_int("hidden_layer_size", 64, 512)
    buffer_size = trial.suggest_int("buffer_size", 50000, 1000000)
    
    
    return train_agent(trial, learning_rate, gamma, batch_size, hidden_layer_size, buffer_size, entropy_coef)

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(optimize_hyperparameters, n_trials=50)

print("Best parameters found:", study.best_params)

with open("best_hyperparameters.yaml", "w") as f:
    yaml.dump(study.best_params, f, default_flow_style=False)
