import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


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
            random_gravity = torch.randn_like(undeformed_vert[0])  # shape (4, 3)
            random_gravity = F.normalize(random_gravity, dim=0) * 9.81
        for t in range(total_time):
            full_traj[t] = undeformed_vert[0] + 0.5 * random_gravity * (t * dt) ** 2

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
