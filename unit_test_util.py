import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import numpy as np




class TrainSimpleTrajData(Dataset):
    def __init__(self, undeformed_vert, train_time_horizon, total_time, n_samples, dt, device="cpu",sim=None, plotting=False):
        super().__init__()
        # sim = Unit_test_sim(batch, n_vert, n_branch, n_edge, pbd_iter, b_DLO_mass, device)
        self.device = device
        self.prev_traj = []
        self.curr_traj = []
        self.targ_traj = []
        self.global_idx = 0
        self.undeformed_vert = undeformed_vert.detach().clone()
        self.plotting = plotting

        for _ in range(n_samples):
            full_traj = sim.generate_preX_trajectory(total_time, dt)
            # generate sliding window segments
            for i in range(total_time - 1 - train_time_horizon):
                prev = full_traj[i: i + train_time_horizon]
                curr = full_traj[i + 1: i + 1 + train_time_horizon]
                targ = full_traj[i + 2: i + 2 + train_time_horizon]

                self.prev_traj.append(prev)
                self.curr_traj.append(curr)
                self.targ_traj.append(targ)


                self.global_idx += 1

        self.prev_traj = torch.stack(self.prev_traj)
        self.curr_traj = torch.stack(self.curr_traj)
        self.targ_traj = torch.stack(self.targ_traj)
        self.save_trajectory_with_undeformed(
            self.curr_traj,  # shape [,n_Sample, T, branch, V, 3]
            self.undeformed_vert,
            idx=self.global_idx,
            save_dir="trajectory_plots",
            title=f"Auto-Saved Trajectory Sample{self.global_idx}"
        )

    def __len__(self):
        return self.curr_traj.shape[0]

    def __getitem__(self, idx):
        return (self.prev_traj[idx].clone().detach(),
                self.curr_traj[idx].clone().detach(),
                self.targ_traj[idx].clone().detach())

    def save_trajectory_with_undeformed(self, trajectory, undeformed_vert, idx=0, save_dir="trajectory_frames",
                                        title="Trajectory Frame"):
        """
        Save a sequence of trajectory snapshots (one per time step) with undeformed reference overlaid.

        Args:
            trajectory: Tensor of shape [n_sample, Time_horizon, n_Branch, Vertices, 3]
            undeformed_vert: Tensor of shape [B, V, 3] or [V, 3]
            idx: Index of the trajectory sample
            save_dir: Directory to save the plot frames
            title: Title for the figure
        """
        # if passed in False, do not plot
        if not self.plotting:
            return

        os.makedirs(save_dir, exist_ok=True)

        n_sample, T, B, V, _ = trajectory.shape
        trajectory = trajectory.cpu()
        undeformed_vert = undeformed_vert.cpu()

        if undeformed_vert.ndim == 2:
            undeformed_vert = undeformed_vert.unsqueeze(0)

        for batch_idx in range(n_sample):
            verts = trajectory[batch_idx]  # [T, B, V, 3]

            for t in range(T):
                print('t', t)
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')

                for b in range(B):
                    print('branch number', b)
                    points = verts[t, b].numpy()  # shape: [V, 3]
                    undeformed_np = undeformed_vert[b].numpy()  # shape: [V, 3]

                    # Create masks to skip [0,0,0] points
                    mask_traj = ~np.all(points == 0, axis=1)
                    mask_undeformed = ~np.all(undeformed_np == 0, axis=1)

                    # Apply mask before plotting
                    points = points[mask_traj]
                    undeformed_np = undeformed_np[mask_undeformed]

                    # Plot only if there are non-zero points
                    if points.shape[0] > 0:
                        ax.plot(points[:, 0], points[:, 1], points[:, 2], alpha=1.0, label=f"Branch {b} at t={t}")
                        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='black', s=10)

                    if undeformed_np.shape[0] > 0:
                        ax.plot(undeformed_np[:, 0], undeformed_np[:, 1], undeformed_np[:, 2],
                                c='green', linestyle='--')
                        ax.scatter(undeformed_np[:, 0], undeformed_np[:, 1], undeformed_np[:, 2],
                                   c='green', s=20, marker='x')

                ax.set_title(f"{title} | Sample {idx}, t={t}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim([-0.5, 1.0])
                ax.set_ylim([-0.5, 1.0])
                ax.set_zlim([-0.5, 0.5])
                ax.view_init(elev=30, azim=-45)
                ax.legend()

                filename = os.path.join(save_dir, f"sample{idx}_t{t:03d}.png")
                plt.tight_layout()
                plt.savefig(filename)
                plt.close()


class EvalSimpleTrajData(Dataset):
    def __init__(self, eval_time_horizon, total_time, n_samples, dt, device="cpu", sim=None,
                 plotting=False):
        super().__init__()
        self.device = device

        self.prev_traj = []
        self.curr_traj = []
        self.targ_traj = []
        self.global_idx = 0 # index for frame saving plots

        for _ in range(n_samples):
            full_traj = sim.generate_preX_trajectory(total_time, dt)

            # take only the window starting at i=0
            prev = full_traj[0:0 + eval_time_horizon]  # [0 .. H-1]
            curr = full_traj[1:1 + eval_time_horizon]  # [1 .. H]
            targ = full_traj[2:2 + eval_time_horizon]  # [2 .. H+1]

            self.prev_traj.append(prev)
            self.curr_traj.append(curr)
            self.targ_traj.append(targ)

        self.prev_traj = torch.stack(self.prev_traj)
        self.curr_traj = torch.stack(self.curr_traj)
        self.targ_traj = torch.stack(self.targ_traj)

    def __len__(self):
        return self.curr_traj.shape[0]

    def __getitem__(self, idx):
        return (self.prev_traj[idx].clone().detach(),
                self.curr_traj[idx].clone().detach(),
                self.targ_traj[idx].clone().detach())

