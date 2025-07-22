import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from Unit_test_sim import Unit_test_sim

import os




class TrainSimpleTrajData(Dataset):
    def __init__(self, undeformed_vert, gravity, time_horizon, total_time, n_samples, dt, device="cpu",sim=None):
        super().__init__()
        # sim = Unit_test_sim(batch, n_vert, n_branch, n_edge, pbd_iter, b_DLO_mass, device)
        self.device = device
        self.time_horizon = time_horizon

        self.prev_traj = []
        self.curr_traj = []
        self.targ_traj = []

        for _ in range(n_samples):

            full_traj = sim.generate_preX_trajectory(total_time, dt)
            # generate sliding window segments
            for i in range(total_time - 2 - time_horizon):
                prev = full_traj[i: i + time_horizon]
                curr = full_traj[i + 1: i + 1 + time_horizon]
                targ = full_traj[i + 2: i + 2 + time_horizon]
                self.prev_traj.append(prev)
                self.curr_traj.append(curr)
                self.targ_traj.append(targ)
                if len(self.curr_traj) > 0:
                    self.save_sample_plot(idx=i, traj_type="curr", title="Auto-Saved Trajectory Sample")

        self.prev_traj = torch.stack(self.prev_traj)
        self.curr_traj = torch.stack(self.curr_traj)
        self.targ_traj = torch.stack(self.targ_traj)



    def __len__(self):
        return self.curr_traj.shape[0]

    def __getitem__(self, idx):
        return (self.prev_traj[idx].clone().detach(),
                self.curr_traj[idx].clone().detach(),
                self.targ_traj[idx].clone().detach())

    def save_sample_plot(self, idx=0, traj_type="curr", title="Trajectory Sample", save_dir="trajectory_plots"):
        """
        Save a single trajectory sample as a 3D plot image.

        Args:
            idx: sample index
            traj_type: one of 'prev', 'curr', or 'targ'
            title: plot title
            save_dir: directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)

        traj_dict = {
            "prev": self.prev_traj,
            "curr": self.curr_traj,
            "targ": self.targ_traj
        }

        assert traj_type in traj_dict, f"traj_type must be one of {list(traj_dict.keys())}"

        trajectory = traj_dict[traj_type][idx]  # shape: [time_horizon, batch * n_branch, n_vert, 3]
        time_horizon, batch_size, n_vert, _ = trajectory.shape

        for b in range(batch_size):
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111, projection='3d')
            for t in range(time_horizon):
                verts = trajectory[t, b].cpu().numpy()
                ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], alpha=t / time_horizon)
            ax.set_title(f"{title} | Sample {idx}, Wire {b}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.tight_layout()

            # Save figure
            filename = f"{traj_type}_sample{idx}_wire{b}.png"
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()

class EvalSimpleTrajData(Dataset):
    def __init__(self, undeformed_vert, gravity, time_horizon, n_samples, dt, device="cpu"):
        super().__init__()
        self.device = device

        self.prev_traj = []
        self.curr_traj = []
        self.targ_traj = []

        for _ in range(n_samples):
            full_traj = torch.zeros(time_horizon + 2, undeformed_vert.shape[0], 3, device=device)
            for t in range(time_horizon + 2):
                random_gravity = gravity + 0.2 * torch.randn(3).to(device)

                full_traj[t] = undeformed_vert + 0.5 * random_gravity * (t * dt) ** 2

            self.prev_traj.append(full_traj[:time_horizon])
            self.curr_traj.append(full_traj[1:time_horizon + 1])
            self.targ_traj.append(full_traj[2:time_horizon + 2])

        self.prev_traj = torch.stack(self.prev_traj)
        self.curr_traj = torch.stack(self.curr_traj)
        self.targ_traj = torch.stack(self.targ_traj)

    def __len__(self):
        return self.curr_traj.shape[0]

    def __getitem__(self, idx):
        return (self.prev_traj[idx].clone().detach(),
                self.curr_traj[idx].clone().detach(),
                self.targ_traj[idx].clone().detach())


