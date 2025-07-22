import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle

from Unit_test_sim import Unit_test_sim  # Your custom simulation class
from unit_test_util import TrainSimpleTrajData, EvalSimpleTrajData
import os

import time

# Hyperparameters
batch = 1
n_vert = 7
n_branch = 1
n_edge = n_vert - 1
pbd_iter = 0
device = "cpu"
total_time = 6  # Total simulation time in seconds
time_horizon = total_time-3
eval_time_horizon = total_time - 2
epochs = 1
dt = 1e-2
n_samples = 1  # Number of trajectories for training/evaluation
timer = 0
experiment_runs = 1
torch.manual_seed(int(time.time()))

# Initialize simulation
# b_DLO_mass = torch.ones(batch, n_vert, device=device)
# b_DLO_mass = torch.tensor([1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 1.0], device=device).repeat(batch, 1)

# === Define Dataset class with previous_positions_traj generation ===
class SimpleTrajectoryDataset(Dataset):
    def __init__(self, target_trajs):
        super().__init__()
        self.target_trajs = target_trajs

    def __len__(self):
        return self.target_trajs.shape[0]

    def __getitem__(self, idx):
        traj = self.target_trajs[idx]  # [time_horizon, n_vert, 3]
        previous_traj = torch.zeros_like(traj)
        previous_traj[1:] = traj[:-1]
        return previous_traj, traj



for run_id in range(experiment_runs):
    print(f"\n========== Run {run_id + 1} / 10 ==========\n")
    rdm_mass = torch.rand(batch, n_vert, device=device) * 0.4 + 0.8

    b_DLO_mass = rdm_mass
    sim = Unit_test_sim(batch, n_vert, n_branch, n_edge, pbd_iter, b_DLO_mass, device)

    sim.train()
    # === Create train/eval datasets ===
    gravity = sim.gravity.detach()
    eval_gravity = gravity * 0.95
    undeformed = sim.undeformed_vert.detach()
    undeformed_vert = undeformed[0]

    train_target_traj = torch.zeros(n_samples, time_horizon, n_vert, 3, device=device)
    eval_target_traj = torch.zeros(n_samples, eval_time_horizon, n_vert, 3, device=device)
    train_dataset = TrainSimpleTrajData(
        undeformed_vert=undeformed_vert,
        gravity=gravity,
        time_horizon=time_horizon,
        total_time=total_time,
        n_samples=n_samples,
        dt=dt,
        device=device,
        sim=sim
    )
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    eval_dataset = EvalSimpleTrajData(
        undeformed_vert=undeformed_vert,
        gravity=gravity * 0.95,
        time_horizon=time_horizon,
        n_samples=n_samples - 2,
        dt=dt,
        device=device
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch, shuffle=False)
    # === Define optimizer and loss ===
    optimizer = optim.SGD([
        sim.undeformed_vert,
        sim.mass_diagonal,
        sim.gravity

    ], lr=1e-3)
    loss_func = nn.MSELoss()

    training_losses = []
    eval_losses = []
    training_epochs = []

    # === Training loop ===
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        batch_count = 0
        # if epoch == 0 and os.path.exists("gravity_only_model.pth"):
        #     sim.load_state_dict(torch.load("gravity_only_model.pth"))


        for previous_positions_traj, current_positions_traj,target_traj in train_loader:
            timer += 1
            print('timer',timer)



            traj_loss, total_loss = sim.iterative_sim(
                time_horizon, current_positions_traj, previous_positions_traj, target_traj, loss_func, dt, timer
            )
            total_loss.backward(retain_graph=True)
            # print("mass_diagonal grad:", sim.mass_diagonal.grad)
            # print("gravity grad:", sim.gravity.grad)
            optimizer.step()
            optimizer.zero_grad()

            # After training


            epoch_train_loss += traj_loss.item()
            batch_count += 1
        # torch.save(sim.state_dict(), 'gravity_only_model.pth')

        avg_train_loss = epoch_train_loss / batch_count
        training_losses.append(avg_train_loss)
        training_epochs.append(epoch)

        # === Evaluation ===
        # Load trained parameters if a checkpoint exists

        # sim.eval()
        # with torch.no_grad():
        #     total_eval_loss = 0.0
        #     eval_positions_traj = undeformed_vert.expand(batch, n_vert, 3).unsqueeze(1).repeat(1, time_horizon, 1, 1)
        #
        #     for previous_eval_traj, _, eval_traj in eval_loader:
        #         eval_positions_traj = torch.zeros_like(eval_traj)
        #
        #         eval_loss, _ = sim.iterative_sim(
        #             time_horizon, eval_positions_traj, previous_eval_traj, eval_traj, loss_func, dt
        #         )
        #         total_eval_loss += eval_loss.item()
        #     avg_eval_loss = total_eval_loss / len(eval_loader)
        #     eval_losses.append(avg_eval_loss)
        #
        # print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Eval Loss: {avg_eval_loss:.6f}")


        # # === Save logs ===
        # with open("training_losses.pkl", "wb") as f:
        #     pickle.dump(training_losses, f)
        # with open("training_epochs.pkl", "wb") as f:
        #     pickle.dump(training_epochs, f)
        # with open("eval_losses.pkl", "wb") as f:
        #     pickle.dump(eval_losses, f)
        #
        # # === Plot loss curve ===
        # plt.figure(figsize=(8, 5))
        # plt.plot(training_epochs, training_losses, marker='o', label="Train Loss")
        # plt.plot(training_epochs, eval_losses, marker='x', label="Eval Loss")
        # plt.title("Training vs. Evaluation Trajectory Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Average Trajectory MSE Loss")
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig("training_vs_eval_loss_plot.png")
        # plt.close()
