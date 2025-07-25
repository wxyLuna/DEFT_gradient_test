import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from Unit_test_sim import Unit_test_sim

import os




class TrainSimpleTrajData(Dataset):
    def __init__(self, undeformed_vert,clamp_selection, gravity, time_horizon, total_time, n_samples, dt, device="cpu",sim=None):
        super().__init__()
        # sim = Unit_test_sim(batch, n_vert, n_branch, n_edge, pbd_iter, b_DLO_mass, device)
        self.device = device
        self.time_horizon = time_horizon

        self.prev_traj = []
        self.curr_traj = []
        self.targ_traj = []
        self.global_idx = 0
        self.undeformed_vert = undeformed_vert.detach().clone()

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


                self.global_idx += 1

        self.prev_traj = torch.stack(self.prev_traj)
        self.curr_traj = torch.stack(self.curr_traj)
        self.targ_traj = torch.stack(self.targ_traj)
        self.save_trajectory_with_undeformed(
            self.curr_traj.squeeze(0),  # shape [T, B=1, V, 3]
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

    def save_trajectory_with_undeformed(self,trajectory, undeformed_vert, idx=0, save_dir="trajectory_plots",
                                        title="Trajectory Sample"):
        """
        Save a trajectory sample with undeformed reference overlaid.

        Args:
            trajectory: Tensor of shape [T, B, V, 3] (time_horizon, batch, vertices, 3D)
            undeformed_vert: Tensor of shape [B, V, 3] or [V, 3]
            idx: Index of the trajectory sample
            save_dir: Directory to save the plot
            title: Title for the figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import os
        import numpy as np

        os.makedirs(save_dir, exist_ok=True)

        T, B, V, _ = trajectory.shape
        trajectory = trajectory.cpu()
        undeformed_vert = undeformed_vert.cpu()
        if undeformed_vert.ndim == 2:  # If shape is [V, 3], expand to [1, V, 3]
            undeformed_vert = undeformed_vert.unsqueeze(0)

        for b in range(B):
            verts = trajectory[:, b]  # shape: [T, V, 3]
            undeformed = undeformed_vert[b]  # shape: [V, 3]

            fig = plt.figure(figsize=(16, 8))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')

            for ax in [ax1, ax2]:
                # Plot time-varying trajectory
                for t in range(T):
                    color_strength = 1
                    points = verts[t].numpy()
                    ax.plot(points[:, 0], points[:, 1], points[:, 2], alpha=color_strength,
                            label=f"t={t}" if t == 0 else "")
                    for v in range(V):
                        x, y, z = points[v]
                        ax.scatter(x, y, z, color='black', s=10)
                        ax.text(x, y, z, f'{v}', fontsize=6)

                # Plot undeformed rest shape in green
                undeformed_np = undeformed.numpy()
                ax.plot(undeformed_np[:, 0], undeformed_np[:, 1], undeformed_np[:, 2], c='green', linestyle='--',
                        label='Undeformed')
                ax.scatter(undeformed_np[:, 0], undeformed_np[:, 1], undeformed_np[:, 2], c='green', s=20, marker='x')

                ax.set_title(f"{title} | Sample {idx}, Wire {b}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim([-0.5, 1.0])
                ax.set_ylim([-0.5, 1.0])
                ax.set_zlim([-0.5, 0.5])
                ax.legend()

            ax1.view_init(elev=0, azim=90)
            ax2.view_init(elev=30, azim=-45)

            filename = os.path.join(save_dir, f"traj_sample{idx}_wire{b}.png")
            plt.tight_layout()
            plt.savefig(filename)
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


