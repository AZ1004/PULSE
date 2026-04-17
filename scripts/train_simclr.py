import wandb

# Initialize a new run
wandb.init(
    project="PULSE-SSL",
    name="initial-architecture-check",
    config={
        "architecture": "ResNet50",
        "dataset": "BUSI",
        "batch_size": 32,
        "learning_rate": 0.0003,
    }
)

# This is where your training loop will eventually live!
# wandb.log({"loss": current_loss}) 

wandb.finish()