import torch
import torch.nn as nn

# Import DEFT functions
from DEFT_func import DEFT_func
from util import rotation_matrix, computeW, computeLengths, computeEdges, visualize_tensors_3d_in_same_plot_no_zeros
from constraints_solver import constraints_enforcement
import pytorch3d.transforms.rotation_conversions
from constraints_enforcement_numba import constraints_enforcement_numba
constraints_numba = constraints_enforcement_numba()

module_dir = "residual_learning_nn"
import sys
import os
sys.path.append(module_dir)

class Unit_test_sim(nn.Module):
    def __init__(self, batch, n_vert, n_branch, n_edge, pbd_iter, b_DLO_mass, device):
        super().__init__()
        self.n_vert = n_vert
        self.n_edge = n_edge
        self.device = device
        self.batch = batch

        rest_vert = torch.tensor([
            [0.893471, -0.133465, 0.018059],
            [0.590795, -0.144219, 0.021808],
            [0.200583, -0.146497, 0.01727],
            [-0.094659, -0.186181, 0.012403]
        ]).unsqueeze(0).repeat(batch, 1, 1).to(device)

        rest_vert = torch.cat((rest_vert[:, :, 0:1], rest_vert[:, :, 2:3], -rest_vert[:, :, 1:2]), dim=-1)
        self.undeformed_vert = nn.Parameter(rest_vert)

        self.gravity = nn.Parameter(torch.tensor((0, 0, -9.81), device=device))
        self.dt = 1e-2

        self.mass_diagonal = nn.Parameter(b_DLO_mass)
        self.mass_matrix = (
            torch.eye(3)
            .unsqueeze(0).unsqueeze(0)
            .repeat(batch, n_vert, 1, 1)
            * self.mass_diagonal.unsqueeze(-1).unsqueeze(-1)
        )  # shape: (batch, n_vert, 3, 3)



        # self.damping = nn.Parameter(torch.tensor(0.1, device=device))  # damping factor

    def External_Force(self, mass_matrix
                       ):
        return torch.matmul(mass_matrix, self.gravity.view(-1, 1)).squeeze(-1)

    def Numerical_Integration(self,mass_matrix,total_force, velocities,positions, dt):

        acc = torch.linalg.solve(mass_matrix, total_force.unsqueeze(-1)).squeeze(-1)
        velocities = velocities + acc * dt
        positions = positions + velocities * dt  # not wrapped in nn.Parameter
        return positions

    def iterative_sim(self, time_horizon, positions_traj, previous_positions_traj,target_traj, loss_func,dt):
        traj_loss_eval = 0.0
        total_loss = 0.0
        total_force = self.External_Force(self.mass_matrix)

        for t in range(time_horizon):
            if t == 0:
                positions = positions_traj[:,t]
                prev_positions = previous_positions_traj[:,t].clone()
                velocities = (positions - prev_positions) / dt
            else:

                prev_positions = positions_old.clone()



            positions = self.Numerical_Integration(self.mass_matrix, total_force, velocities,
                                                               positions, dt)
            velocities = (positions - prev_positions) / dt

            gt_positions = target_traj[:, t]
            gt_velocities = (target_traj[:, t] - positions_traj[:, t]) / dt
            step_loss_pos = loss_func(positions, gt_positions)
            step_loss_vel = loss_func(velocities, gt_velocities)

            traj_loss_eval += step_loss_pos
            total_loss += (step_loss_pos + step_loss_vel)


            # positions_traj[:, t] = positions.detach()
            positions_old = positions.clone()


        return traj_loss_eval, total_loss
