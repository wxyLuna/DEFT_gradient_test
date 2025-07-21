import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



class TrainSimpleTrajData(Dataset):
    def __init__(self, undeformed_vert, gravity, time_horizon, total_time, n_samples, dt, device="cpu"):
        super().__init__()
        self.device = device
        self.time_horizon = time_horizon

        self.prev_traj = []
        self.curr_traj = []
        self.targ_traj = []

        for _ in range(n_samples):
            full_traj = torch.zeros(total_time, undeformed_vert.shape[0], 3, device=device)

        for t in range(total_time):
            print('total_time', total_time)
            random_gravity = torch.rand(undeformed_vert.shape[0], 3) * torch.tensor([[1, 1, 1]])  # shape (4, 3)
            random_gravity = F.normalize(random_gravity, dim=0) * 9.81
            print("random_gravity", random_gravity)
            full_traj[t] = undeformed_vert[0] + 0.5 * random_gravity * (t * dt) ** 2
            print("full_traj", full_traj[t])

            # generate sliding window segments
            for i in range(total_time - 2 - time_horizon):
                prev = full_traj[i: i + time_horizon]
                curr = full_traj[i + 1: i + 1 + time_horizon]
                targ = full_traj[i + 2: i + 2 + time_horizon]
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

def visualize_tensors_3d_in_same_plot(
        parent_clamped_selection, tensor_1, tensor_2, ith, test_data_index,
        clamp_parent, clamp_child1, clamp_child2,
        parent_fix_point_flat2, child1_fix_point_flat, child2_fix_point_flat,
        additional_tensor_1, additional_tensor_2, additional_tensor_3, i_eval_batch, vis_type
):
    """
    Plots multiple 3D tensor datasets in a single figure with two 3D subplots (two different view angles).
    Zeros are filtered out (ignored), and certain indices from parent_clamped_selection are highlighted.

    Args:
        parent_clamped_selection (torch.Tensor): Indices in tensor_1 (the main rod) that are clamped or special.
        tensor_1 (torch.Tensor): A main set of 3D points (e.g., predicted rod).
        tensor_2 (torch.Tensor): Possibly multiple sets of 3D points to overlay (e.g., other rods).
        additional_tensor_1, additional_tensor_2, additional_tensor_3: Extra sets of points to overlay (e.g. ground truth).
        clamp_parent, clamp_child1, clamp_child2 (bool): Flags that decide if fix points are plotted.
        parent_fix_point_flat2, child1_fix_point_flat, child2_fix_point_flat (torch.Tensor or None):
            The fix/clamp points to highlight if clamps are used.
        i_eval_batch (int): ID for saving directory path.
        vis_type (str): Directory subfolder for saving the results.

    Returns:
        None. Saves a .png file of the 3D plot.
    """

    def filter_non_zero_points(tensor):
        # Filters out rows that are all zeros in the last dimension
        non_zero_mask = torch.any(tensor != 0, dim=-1)
        return tensor[non_zero_mask]

    # Create 2 subplots (two views) in 3D
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('View 1')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('View 2')

    # Core plotting function
    def plot_tensors(ax):
        N = tensor_1.shape[0]

        # Convert negative indices to positive with modulo, then remove duplicates
        selected_indices = parent_clamped_selection % N
        selected_indices = np.unique(selected_indices)
        total_indices = np.arange(N)
        non_selected_indices = np.setdiff1d(total_indices, selected_indices)

        # Identify negative indices (originally <0)
        negative_indices = parent_clamped_selection[parent_clamped_selection < 0] % N
        negative_indices = np.unique(negative_indices)
        # Positive indices
        positive_indices = np.setdiff1d(selected_indices, negative_indices)

        # Plot additional_tensor_1 (likely ground truth)
        additional_tensor_1_filtered = filter_non_zero_points(additional_tensor_1)
        if additional_tensor_1_filtered.size(0) > 0:
            ax.scatter(
                additional_tensor_1_filtered[:, 0].detach().cpu().numpy(),
                additional_tensor_1_filtered[:, 1].detach().cpu().numpy(),
                additional_tensor_1_filtered[:, 2].detach().cpu().numpy(),
                c='black', marker='o', s=30, label='Ground Truth'
            )
        # # Plot additional_tensor_2
        # additional_tensor_2_filtered = filter_non_zero_points(additional_tensor_2)
        # if additional_tensor_2_filtered.size(0) > 0:
        #     ax.scatter(
        #         additional_tensor_2_filtered[:, 0].detach().cpu().numpy(),
        #         additional_tensor_2_filtered[:, 1].detach().cpu().numpy(),
        #         additional_tensor_2_filtered[:, 2].detach().cpu().numpy(),
        #         c='black', marker='o', alpha=1.0, s=30
        #     )
        # # Plot additional_tensor_3
        # additional_tensor_3_filtered = filter_non_zero_points(additional_tensor_3)
        # if additional_tensor_3_filtered.size(0) > 0:
        #     ax.scatter(
        #         additional_tensor_3_filtered[:, 0].detach().cpu().numpy(),
        #         additional_tensor_3_filtered[:, 1].detach().cpu().numpy(),
        #         additional_tensor_3_filtered[:, 2].detach().cpu().numpy(),
        #         c='black', marker='o', alpha=1.0, s=30
        #     )

        # Plot non-selected points from tensor_1
        tensor_1_non_selected = tensor_1[non_selected_indices]
        tensor_1_non_selected = filter_non_zero_points(tensor_1_non_selected)
        if tensor_1_non_selected.size(0) > 0:
            ax.scatter(
                tensor_1_non_selected[:, 0].detach().cpu().numpy(),
                tensor_1_non_selected[:, 1].detach().cpu().numpy(),
                tensor_1_non_selected[:, 2].detach().cpu().numpy(),
                c='blue', marker='o', alpha=1.0, s=30, label='Prediction'
            )

        # Plot selected points with positive indices
        tensor_1_positive = tensor_1[positive_indices]
        tensor_1_positive = filter_non_zero_points(tensor_1_positive)
        if tensor_1_positive.size(0) > 0:
            ax.scatter(
                tensor_1_positive[:, 0].detach().cpu().numpy(),
                tensor_1_positive[:, 1].detach().cpu().numpy(),
                tensor_1_positive[:, 2].detach().cpu().numpy(),
                c='red', marker='o', alpha=1.0, s=40, label='Clamped Points'
            )

        # Plot selected points with negative indices
        tensor_1_negative = tensor_1[negative_indices]
        tensor_1_negative = filter_non_zero_points(tensor_1_negative)
        if tensor_1_negative.size(0) > 0:
            ax.scatter(
                tensor_1_negative[:, 0].detach().cpu().numpy(),
                tensor_1_negative[:, 1].detach().cpu().numpy(),
                tensor_1_negative[:, 2].detach().cpu().numpy(),
                c='red', marker='o', alpha=1.0, s=40
            )

        # Plot each row in tensor_2
        # for i in range(tensor_2.shape[0]):
        #     filtered_tensor_2 = filter_non_zero_points(tensor_2[i])
        #     if filtered_tensor_2.size(0) > 0:
        #         ax.scatter(
        #             filtered_tensor_2[:, 0].detach().cpu().numpy(),
        #             filtered_tensor_2[:, 1].detach().cpu().numpy(),
        #             filtered_tensor_2[:, 2].detach().cpu().numpy(),
        #             c='blue', alpha=1.0, s=30, marker='o'
        #         )

        # # If child rods are clamped, plot the fix points
        # if clamp_child1 and child1_fix_point_flat is not None:
        #     points = child1_fix_point_flat[0]
        #     filtered_points = filter_non_zero_points(points)
        #     if filtered_points.size(0) > 0:
        #         ax.scatter(
        #             filtered_points[:, 0].detach().cpu().numpy(),
        #             filtered_points[:, 1].detach().cpu().numpy(),
        #             filtered_points[:, 2].detach().cpu().numpy(),
        #             c='red', s=40, marker='o', label='Child1 Fix Points'
        #         )
        # if clamp_child2 and child2_fix_point_flat is not None:
        #     points = child2_fix_point_flat[0]
        #     filtered_points = filter_non_zero_points(points)
        #     if filtered_points.size(0) > 0:
        #         ax.scatter(
        #             filtered_points[:, 0].detach().cpu().numpy(),
        #             filtered_points[:, 1].detach().cpu().numpy(),
        #             filtered_points[:, 2].detach().cpu().numpy(),
        #             c='red', s=40, marker='o', label='Child2 Fix Points'
        #         )

        # Configure plot
        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(-0.5, 1.0)
        ax.set_zlim(-0.25, 1.25)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    # Plot on both subplots
    plot_tensors(ax1)
    plot_tensors(ax2)

    # Adjust viewpoints
    ax1.view_init(elev=0, azim=90)
    ax2.view_init(elev=30, azim=-45)

    # Save the figure
    dir_path = Path(f"sanity_check/{vis_type}/{i_eval_batch}")
    dir_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(f"sanity_check/{vis_type}/{i_eval_batch}/{ith}.png")
    plt.close(fig)
