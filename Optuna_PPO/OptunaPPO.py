import optuna
import subprocess
import yaml
import re

def train_agent(trial, learning_rate, gamma, batch_size, hidden_layer_size, buffer_size, epochs):
    config = {
        "behaviors": {
            "MoveToGoal": {
                "trainer_type": "ppo",
                "hyperparameters": {
                    "batch_size": batch_size,
                    "buffer_size": buffer_size,
                    "learning_rate": learning_rate,
                    "beta": 5.0e-3,
                    "epsilon": 0.2,
                    "lambd": 0.95,
                    "num_epoch": epochs,
                    "learning_rate_schedule": "linear"
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
                "summary_freq": 50000
            }
        }
    }
    
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    #Run mlagents command
    result = subprocess.run([
        "mlagents-learn", "config.yaml", "--run-id=optuna_run",
        "--env=../UnityBuilds/OptunaBuild.app",
        "--force", "--no-graphics",
        "--width=1920", "--height=1080"
    ], capture_output=True, text=True)
    
    reward = None
    rewards = []
    step = 0
    
    #Scan output from mlagents command and extract rewards
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
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    batch_size = trial.suggest_int("batch_size", 512, 5120, step=512)
    hidden_layer_size = trial.suggest_int("hidden_layer_size", 32, 512, step=32)
    buffer_size = trial.suggest_int("buffer_size", 2048, 40960, step=2048)
    epochs = trial.suggest_int("epochs", 5, 50, step=5)
    
    return train_agent(trial, learning_rate, gamma, batch_size, hidden_layer_size, buffer_size, epochs)

#Create and run optuna study
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(optimize_hyperparameters, n_trials=50)

print("Best parameters found:", study.best_params)

with open("best_hyperparameters.yaml", "w") as f:
    yaml.dump(study.best_params, f, default_flow_style=False)

