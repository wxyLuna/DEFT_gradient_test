import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Unit_test_sim import Unit_test_sim
import pickle
import matplotlib.pyplot as plt

# Hyperparameters
batch = 1
n_vert = 4
n_branch = 1
n_edge = n_vert - 1
pbd_iter = 0
device = "cpu"
b_DLO_mass = torch.ones(batch, n_vert, device=device)
time_horizon = 20
epochs = 30
dt = 1e-2

# Initialize simulation
sim = Unit_test_sim(batch, n_vert, n_branch, n_edge, pbd_iter, b_DLO_mass, device)
sim.train()

# Create target trajectory (falling under gravity)
target_traj = torch.zeros(batch, time_horizon, n_vert, 3, device=device)
for t in range(time_horizon):
    target_traj[:, t] = sim.undeformed_vert.detach() + 0.5 * sim.gravity.detach() * (t * dt) ** 2

# Define optimizer (only train select parameters)
optimizer = optim.Adam([
    sim.undeformed_vert,
    sim.mass_diagonal

], lr=1e-3)

loss_func = nn.MSELoss()
training_losses = []
eval_losses = []
training_epochs = []

# Create eval trajectory with perturbed gravity
eval_gravity = sim.gravity.detach() * 0.95
eval_target_traj = torch.zeros(batch, time_horizon, n_vert, 3, device=device)
for t in range(time_horizon):
    eval_target_traj[:, t] = sim.undeformed_vert.detach() + 0.5 * eval_gravity * (t * dt) ** 2

for epoch in range(epochs):
    optimizer.zero_grad()

    # Reset state
    sim.positions.data = sim.undeformed_vert.data.clone()
    sim.velocities.data.zero_()
    positions_traj = torch.zeros(batch, time_horizon, n_vert, 3, device=device)

    # Forward and backward pass
    traj_loss, total_loss = sim.iterative_sim(time_horizon, positions_traj, target_traj, loss_func)
    total_loss.backward(retain_graph=True)
    optimizer.step()

    training_losses.append(traj_loss.item() / time_horizon)
    training_epochs.append(epoch)

    # Evaluation
    with torch.no_grad():
        sim.positions.data = sim.undeformed_vert.data.clone()
        sim.velocities.data.zero_()
        eval_positions_traj = torch.zeros(batch, time_horizon, n_vert, 3, device=device)
        eval_loss, _ = sim.iterative_sim(time_horizon, eval_positions_traj, eval_target_traj, loss_func)
        eval_losses.append(eval_loss.item() / time_horizon)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {training_losses[-1]:.6f} | Eval Loss: {eval_losses[-1]:.6f}")

    # Save logs
    with open("training_losses.pkl", "wb") as f:
        pickle.dump(training_losses, f)
    with open("training_epochs.pkl", "wb") as f:
        pickle.dump(training_epochs, f)
    with open("eval_losses.pkl", "wb") as f:
        pickle.dump(eval_losses, f)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(training_epochs, training_losses, marker='o', label="Train Loss")
    plt.plot(training_epochs, eval_losses, marker='x', label="Eval Loss")
    plt.title("Training vs. Evaluation Trajectory Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Trajectory MSE Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_vs_eval_loss_plot.png")
    plt.close()
