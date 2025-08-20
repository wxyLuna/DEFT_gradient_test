import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
from util import clamp_index,index_init, DEFT_initialization, construct_b_DLOs, visualize_tensors_3d_in_same_plot_no_zeros

from Unit_test_sim import Unit_test_sim  # Your custom simulation class
from unit_test_util import TrainSimpleTrajData, EvalSimpleTrajData
import os

import time

# Hyperparameters -- refer to DEFT_train BDLO 1
batch = 1
n_parent_vertices = 13
n_child1_vertices = 5
n_child2_vertices = 4
n_vert = n_parent_vertices
n_children_vertices = (n_child1_vertices, n_child2_vertices)
n_branch = 3
n_edge = n_vert - 1
pbd_iter = 0
device = "cpu"
total_time = 40 # Total simulation time in seconds
train_time_horizon = total_time-2
eval_time_horizon = total_time - 2
epochs = 1
dt = 1e-2
n_samples = 1  # Number of trajectories(batches) for training/evaluation
timer = 0
experiment_runs = 1
torch.manual_seed(int(time.time()))


rest_vert = torch.tensor([[[-0.6790, -0.6355, -0.5595, -0.4539, -0.3688, -0.2776, -0.1857,
                                          -0.0991, 0.0102, 0.0808, 0.1357, 0.2081, 0.2404, -0.4279,
                                          -0.4880, -0.5394, -0.5559, 0.0698, 0.0991, 0.1125]],
                                        [[0.0035, -0.0066, -0.0285, -0.0349, -0.0704, -0.0663, -0.0744,
                                          -0.0957, -0.0702, -0.0592, -0.0452, -0.0236, -0.0134, -0.0813,
                                          -0.1233, -0.1875, -0.2178, -0.1044, -0.1858, -0.2165]],
                                        [[0.0108, 0.0104, 0.0083, 0.0104, 0.0083, 0.0145, 0.0133,
                                          0.0198, 0.0155, 0.0231, 0.0199, 0.0154, 0.0169, 0.0160,
                                          0.0153, 0.0090, 0.0121, 0.0205, 0.0155, 0.0148]]]).permute(1, 2, 0)

rest_vert = torch.cat((rest_vert[:, :, 0:1], rest_vert[:, :, 2:3], -rest_vert[:, :, 1:2]), dim=-1)
parent_vertices_undeform = rest_vert[:, :n_parent_vertices]
child1_vertices_undeform = rest_vert[:, n_parent_vertices: n_parent_vertices + n_children_vertices[0] - 1]
child2_vertices_undeform = rest_vert[:, n_parent_vertices + n_children_vertices[0] - 1:]

n_parent_vertices = 13
n_child1_vertices = 5
n_child2_vertices = 4
parent_clamped_selection = torch.tensor((0, 1, -2, -1))
child1_clamped_selection = torch.tensor((2))
child2_clamped_selection = torch.tensor((2))
parent_mass_scale = 1.
parent_moment_scale = 10.
moment_ratio = 0.1
children_moment_scale = (0.5, 0.5)
children_mass_scale = (1, 1)
rigid_body_coupling_index = [4, 8]
clamp_parent = True
clamp_child1 = False
clamp_child2 = False
b_DLOs_vertices_undeform_untransform, _ = construct_b_DLOs(
        batch,
        rigid_body_coupling_index,
        n_parent_vertices,
        n_children_vertices,
        n_branch,
        parent_vertices_undeform,
        parent_vertices_undeform,
        child1_vertices_undeform,
        child1_vertices_undeform,
        child2_vertices_undeform,
        child2_vertices_undeform
    )
b_DLOs_vertices_undeform_transform = torch.zeros_like(b_DLOs_vertices_undeform_untransform)
b_DLOs_vertices_undeform_transform[:, :, :, 0] = -b_DLOs_vertices_undeform_untransform[:, :, :, 2]
b_DLOs_vertices_undeform_transform[:, :, :, 1] = -b_DLOs_vertices_undeform_untransform[:, :, :, 0]
b_DLOs_vertices_undeform_transform[:, :, :, 2] = b_DLOs_vertices_undeform_untransform[:, :, :, 1]
b_undeformed_vert = b_DLOs_vertices_undeform_transform[0].view(n_branch, -1, 3)
clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(batch, parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
                                         n_branch, n_vert, clamp_parent, clamp_child1, clamp_child2) # hardcoded clamped index for the first vertex
index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(
        rigid_body_coupling_index,n_branch)
b_DLO_mass, parent_MOI, children_MOI, parent_rod_orientation, children_rod_orientation, b_nominal_length = DEFT_initialization(
        parent_vertices_undeform,
        child1_vertices_undeform,
        child2_vertices_undeform,
        n_branch,
        n_parent_vertices,
        n_children_vertices,
        rigid_body_coupling_index,
        parent_mass_scale,
        parent_moment_scale,
        children_moment_scale,
        children_mass_scale,
        moment_ratio
    )
damping = nn.Parameter(torch.tensor((2.5, 2.5, 2.5), device=device))

##for rest_vert randomization
rdm_scale = 0.03 # Scale for randomizing rest vertices
mass_low, mass_high = 0.8, 1.2 # Mass range for randomization
plotting = False # if True, saves trajectory frames
randomize_rest = False  # if True, jitter rest-vertices & mass

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
    # randomize rest vertices slightly for each experiment run
    if randomize_rest:

        rdm_vec = torch.rand(batch, n_branch, n_vert, 3, device=device)
        rdm_vec = rdm_vec / rdm_vec.norm(dim=-1, keepdim=True) * rdm_scale
        b_undeformed_vert = b_undeformed_vert + rdm_vec
        b_DLO_mass = (mass_high - mass_low) * torch.rand(batch, n_vert, device=device) + mass_low
    else:
        b_undeformed_vert = b_undeformed_vert.clone()
        b_DLO_mass = b_DLO_mass.clone()

    sim = Unit_test_sim(batch,
                        n_vert,
                        n_branch,
                        n_children_vertices,
                        n_edge,
                        b_DLO_mass,
                        b_undeformed_vert,
                        parent_MOI,
                        children_MOI,
                        clamped_index,
                        rigid_body_coupling_index,
                        parent_MOI_index1,
                        parent_MOI_index2,
                        parent_clamped_selection,
                        child1_clamped_selection,
                        child2_clamped_selection,
                        clamp_parent,
                        clamp_child1,
                        clamp_child2,
                        damping,
                        device)

    sim.train()
    # === Create train/eval datasets ===
    gravity = sim.gravity.detach()
    eval_gravity = gravity * 0.95
    undeformed = sim.undeformed_vert.detach()
    undeformed_vert_for_data_generation = undeformed

    train_target_traj = torch.zeros(n_samples, eval_time_horizon, n_vert, 3, device=device)
    eval_target_traj = torch.zeros(n_samples, eval_time_horizon, n_vert, 3, device=device)
    train_dataset = TrainSimpleTrajData(
        undeformed_vert=undeformed_vert_for_data_generation,
        train_time_horizon=train_time_horizon,
        total_time=total_time,
        n_samples=n_samples,
        dt=dt,
        device=device,
        sim=sim,
        plotting=plotting
    )
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    print("train_dataset length:", len(train_dataset))

    eval_dataset = EvalSimpleTrajData(
        eval_time_horizon=eval_time_horizon,
        total_time=total_time,
        n_samples=n_samples,
        dt=dt,
        device=device,
        sim=sim
    )
    # eval_loader = DataLoader(eval_dataset, batch_size=batch, shuffle=False)
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

        for previous_positions_traj, current_positions_traj,target_traj in train_loader:
            timer += 1
            print('timer',timer)




            traj_loss, total_loss = sim.iterative_sim(
                train_time_horizon, current_positions_traj, previous_positions_traj, target_traj, loss_func, dt, timer
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
