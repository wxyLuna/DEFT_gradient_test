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
from constraints_solver import constraints_enforcement
from util import rotation_matrix, computeW, computeLengths, computeEdges, visualize_tensors_3d_in_same_plot_no_zeros
import gradient_saver

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
        self.b_undeformed_vert = rest_vert.clone()
        self.zero_mask = torch.all(self.b_undeformed_vert[:, 1:] == 0, dim=-1)
        self.zero_mask_num = 1 - self.zero_mask.repeat(batch, 1).to(torch.uint8)
        self.m_restEdgeL, self.m_restRegionL = computeLengths(
            computeEdges(self.b_undeformed_vert.clone(), self.zero_mask)
        )
        self.batched_m_restEdgeL = self.m_restEdgeL.repeat(self.batch, 1, 1).view(-1, n_edge)
        self.undeformed_vert = nn.Parameter(rest_vert)
        ## for storing the old gradients from inextensibility enforcement
        self.bkgrad = gradient_saver.BackwardGradientIC(self.batch *n_branch, n_vert)

        self.gravity = nn.Parameter(torch.tensor((0, 0, -9.81), device=device))
        self.dt = 1e-2

        self.mass_diagonal = nn.Parameter(b_DLO_mass)
        self.mass_matrix = (
            torch.eye(3)
            .unsqueeze(0).unsqueeze(0)
            .repeat(batch, n_vert, 1, 1)
            * self.mass_diagonal.unsqueeze(-1).unsqueeze(-1)
        )  # shape: (batch, n_vert, 3, 3)
        mass_scale1 = self.mass_matrix[:, 1:] @ torch.linalg.pinv(self.mass_matrix[:, 1:] + self.mass_matrix[:, :-1])
        mass_scale2 = self.mass_matrix[:, :-1] @ torch.linalg.pinv(self.mass_matrix[:, 1:] + self.mass_matrix[:, :-1])
        self.mass_scale = torch.cat((mass_scale1, -mass_scale2), dim=1).view(-1, self.n_edge, 3, 3)
        self.constraints_enforcement = constraints_enforcement(n_branch)
        self.clamped_index = torch.tensor([[1.0, 0.0, 0.0, 1.0]]) # hardcoded clamped index for the first vertex
        self.inext_scale = self.clamped_index * (1e20)+1 # clamped points does not move
        self.n_branch=n_branch




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
                # print('at time step', t)
                positions = positions_traj[:,t]
                prev_positions = previous_positions_traj[:,t].clone()
                velocities = (positions - prev_positions) / dt
            else:
                # print('else at time step', t)

                prev_positions = positions_old.clone()



            positions = self.Numerical_Integration(self.mass_matrix, total_force, velocities,
                                                               positions, dt)


            positions_ICE, grad_per_ICitr = self.constraints_enforcement.Inextensibility_Constraint_Enforcement(
                self.batch,
                positions,
                self.batched_m_restEdgeL, ## change to nominal length
                self.mass_matrix,
                self.clamped_index,
                self.inext_scale,
                self.mass_scale,
                self.zero_mask_num,
                self.b_undeformed_vert,
                self.bkgrad,
                self.n_branch
            )

            self.bkgrad.grad_DX_X = grad_per_ICitr.grad_DX_X
            self.bkgrad.grad_DX_Xinit = grad_per_ICitr.grad_DX_Xinit
            self.bkgrad.grad_DX_M = grad_per_ICitr.grad_DX_M
            # print('grad_DX_X', self.bkgrad.grad_DX_X)
            # print('grad_DX_Xinit', self.bkgrad.grad_DX_Xinit)
            # print('grad_DX_M', self.bkgrad.grad_DX_M)

            d_positions = torch.tensor([[[0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0]]],
                                        device=self.device)
            d_mass_matrix = torch.tensor([[[[1e-4, 0.0, 0.0],
                                            [0.0, 1e-4, 0.0],
                                            [0.0, 0.0, 1e-4]],
                                           [[0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]],
                                           [[0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]],
                                           [[0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]]]],
                                           device=self.device)
            
            positions_negative = positions - d_positions
            positions_positive = positions + d_positions
            mass_negative = self.mass_matrix - d_mass_matrix
            mass_positive = self.mass_matrix + d_mass_matrix

            positions_ICE_neg, _ = self.constraints_enforcement.Inextensibility_Constraint_Enforcement(
                self.batch,
                positions_negative,
                self.batched_m_restEdgeL,  # change to nominal length
                mass_negative,
                self.clamped_index,
                self.inext_scale,
                self.mass_scale,
                self.zero_mask_num,
                self.b_undeformed_vert,
                self.bkgrad,
                self.n_branch
            )
            print('positions_ICE_neg',positions_ICE_neg)
            print('positions_negative',positions_negative)

            delta_positions_ICE_neg = (positions_ICE_neg - positions_negative)

            positions_ICE_pos, _ = self.constraints_enforcement.Inextensibility_Constraint_Enforcement(
                self.batch,
                positions_positive,
                self.batched_m_restEdgeL,  # change to nominal length
                mass_positive,
                self.clamped_index,
                self.inext_scale,
                self.mass_scale,
                self.zero_mask_num,
                self.b_undeformed_vert,
                self.bkgrad,
                self.n_branch
            )
            print('positions_ICE_pos', positions_ICE_pos)
            print('positions_positive', positions_positive)

            delta_positions_ICE_pos = (positions_ICE_pos - positions_positive)

            d_delta_positions_ICE = (delta_positions_ICE_pos - delta_positions_ICE_neg) / 2

            d_mass_matrix_block = torch.block_diag(*d_mass_matrix[0]).unsqueeze(0)  # → (1, 12, 12)
            d_positions = d_positions.detach().cpu().numpy()
            d_mass_matrix_block = d_mass_matrix_block.detach().cpu().numpy()
            analytical_d_delta_positions_ICE = self.bkgrad.grad_DX_X @ d_positions.reshape(1,-1,1) + d_mass_matrix_block @ self.bkgrad.grad_DX_M
            #mass_matrix expand to (1,12,12)diagonally position flatten (1,12,1)
            print('d_delta_positions_ICE', d_delta_positions_ICE)
            print('analytical_d_delta_positions_ICE', analytical_d_delta_positions_ICE)

            velocities = (positions_ICE - prev_positions) / dt


            gt_positions = target_traj[:, t]
            gt_velocities = (target_traj[:, t] - positions_traj[:, t]) / dt
            step_loss_pos = loss_func(positions_ICE, gt_positions)
            step_loss_vel = loss_func(velocities, gt_velocities)

            traj_loss_eval += step_loss_pos
            total_loss += (step_loss_pos + step_loss_vel)


            # positions_traj[:, t] = positions.detach()
            positions_old = positions_ICE.clone()


        return traj_loss_eval, total_loss
