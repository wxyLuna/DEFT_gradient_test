import torch
import torch.nn as nn

# Import DEFT functions
from DEFT_func import DEFT_func
from util import rotation_matrix, computeW, computeLengths, computeEdges, visualize_tensors_3d_in_same_plot_no_zeros
from constraints_solver import constraints_enforcement
import pytorch3d.transforms.rotation_conversions
from constraints_enforcement_numba import constraints_enforcement_numba
constraints_numba = constraints_enforcement_numba()
import scipy

module_dir = "residual_learning_nn"
import sys
import os
sys.path.append(module_dir)
from constraints_solver import constraints_enforcement
from util import rotation_matrix, computeW, computeLengths, computeEdges, visualize_tensors_3d_in_same_plot_no_zeros
import gradient_saver
import numpy as np

class Unit_test_sim(nn.Module):
    def __init__(self, batch, n_vert, n_branch, n_edge, pbd_iter, b_DLO_mass, device):
        super().__init__()
        self.n_vert = n_vert
        self.n_edge = n_edge
        self.device = device
        self.batch = batch

        # rest_vert = torch.tensor([
        #     [0.893471, -0.133465, 0.018059],
        #     [0.590795, -0.144219, 0.021808],
        #     [0.200583, -0.146497, 0.01727],
        #     [-0.094659, -0.186181, 0.012403]
        # ]).unsqueeze(0).repeat(batch, 1, 1).to(device)
        rest_vert = torch.tensor([
            [0.893471, -0.133465, 0.018059],
            # [0.880771, -0.119666, 0.017733],
            [0.791946, -0.084258, 0.009944],
            # [0.680462, -0.102366, 0.018528],
            [0.590795, -0.144219, 0.021808],
            # [0.494905, -0.156384, 0.017816],
            [0.396916, -0.143114, 0.021549],
            # [0.299291, -0.148755, 0.014955],
            [0.200583, -0.146497, 0.01727],
            # [0.09586, -0.142385, 0.016456],
            [-0.000782, -0.147084, 0.016081],
            # [-0.071514, -0.17382, 0.015446]
            [-0.094659, -0.186181, 0.012403]

        ]).unsqueeze(0).repeat(batch, 1, 1).to(device)

        rest_vert = torch.cat((rest_vert[:, :, 0:1], rest_vert[:, :, 2:3], -rest_vert[:, :, 1:2]), dim=-1)
        self.b_undeformed_vert = rest_vert.clone()
        self.zero_mask = torch.all(self.b_undeformed_vert[:, 1:] == 0, dim=-1)
        self.zero_mask_num = 1 - self.zero_mask.repeat(batch, 1).to(torch.uint8)
        self.d_positions_init = torch.tensor([[[0.0, 0.0, 0.0],

                                 [0e-2, 0e-2, 0e-2],

                                 [2e-3, 0e-2, 1e-3],
                                 [0e-2, 0e-4, 0e-4],

                                 [0e-3, 0e-3, 0e-3],
                                 [0e-3, 0e-3, 0e-3],
                                 [0.0, 0.0, 0.0]]])
        self.m_restEdgeL, self.m_restRegionL = computeLengths(
            computeEdges(self.b_undeformed_vert.clone(), self.zero_mask)
        )
        self.batched_m_restEdgeL = self.m_restEdgeL.repeat(self.batch, 1, 1).view(-1, n_edge)

        self.m_restEdgeL_pos, self.m_restRegionL_pos = computeLengths(
            computeEdges(self.b_undeformed_vert.clone()+self.d_positions_init.clone(), self.zero_mask)
        )
        self.batched_m_restEdgeL_pos = self.m_restEdgeL_pos.repeat(self.batch, 1, 1).view(-1, n_edge)

        self.m_restEdgeL_neg, self.m_restRegionL_neg = computeLengths(
            computeEdges(self.b_undeformed_vert.clone() - self.d_positions_init.clone(), self.zero_mask)
        )
        self.batched_m_restEdgeL_neg = self.m_restEdgeL_neg.repeat(self.batch, 1, 1).view(-1, n_edge)
        self.undeformed_vert = nn.Parameter(rest_vert)
        ## for storing the old gradients from inextensibility enforcement
        self.bkgrad = gradient_saver.BackwardGradientIC(self.batch *n_branch, n_vert)
        self.bkgrad_neg = gradient_saver.BackwardGradientIC(self.batch * n_branch, n_vert)
        self.bkgrad_pos = gradient_saver.BackwardGradientIC(self.batch * n_branch, n_vert)

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
        self.clamped_index = torch.tensor([[1.0, 1.0,0,0,0, 1.0, 1.0]]) # hardcoded clamped index for the first vertex
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

        self.bkgrad.reset(self.batch, self.n_vert)

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
                                                               prev_positions, dt)

            positions_pre_constraint = positions.clone()
            b_undeformed_vert_pre_constraint = self.b_undeformed_vert.clone()


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
            # print('positions_ICE', positions_ICE)

            d_positions = np.array([[[0.0, 0.0, 0.0],

                                 [0e-2, 0e-2, 0e-2],

                                 [0e-2, 0e-2, 0e-2],
                                 [4e-3, 0e-3, 1e-3],

                                 [0e-3, 0e-3, 0e-3],
                                 [0e-3, 0e-3, 0e-3],
                                 [0.0, 0.0, 0.0]]])

            # d_mass = np.array([[[0.0],
            #                            [0.5e-2],
            #                            [2e-2],
            #                            [4e-2],
            #                            [4e-2],
            #                            [0.0],
            #                            [4e-2]]])
            d_mass = np.array([[[0.0],
                                [0.0],
                                [3e-3],
                                [2e-3],
                                [0.0],
                                [0.0],
                                [0.0]]])
            d_positions_init = self.d_positions_init.detach().cpu().numpy()


            # print('self.bkgrad.grad_DX_X', self.bkgrad.grad_DX_X)

            analytical_d_delta_positions_ICE = (np.matmul(self.bkgrad.grad_DX_Xinit, d_positions_init.reshape(1, -1, 1))+np.matmul(self.bkgrad.grad_DX_X, d_positions.reshape(1, -1, 1)) + np.matmul(self.bkgrad.grad_DX_M, d_mass))
            analytical_d_delta_positions_ICE = analytical_d_delta_positions_ICE.reshape(self.bkgrad.grad_DX_X.shape[0], -1, 3)

            d_positions= torch.from_numpy(d_positions).to(self.device)
            d_mass_matrix = np.eye(3)[None, :, :] * d_mass.squeeze()[:, None, None]
            d_mass_matrix = torch.from_numpy(d_mass_matrix).to(self.device)

            
            positions_negative = positions_pre_constraint - d_positions


            positions_negative_input = positions_negative.clone()

            positions_positive = positions_pre_constraint + d_positions
            positions_positive_input = positions_positive.clone()
            mass_negative = self.mass_matrix - d_mass_matrix
            mass_negative_input = mass_negative.clone()

            mass_scale1_neg = mass_negative_input[:, 1:] @ torch.linalg.pinv(mass_negative_input[:, 1:] + mass_negative_input[:, :-1])
            mass_scale2_neg = mass_negative_input[:, :-1] @ torch.linalg.pinv(mass_negative_input[:, 1:] + mass_negative_input[:, :-1])
            mass_scale_neg = torch.cat((mass_scale1_neg, -mass_scale2_neg), dim=1).view(-1, self.n_edge, 3, 3)

            mass_positive = self.mass_matrix + d_mass_matrix
            # print('mass_positive',mass_positive)
            mass_positive_input = mass_positive.clone()
            mass_scale1_pos = mass_positive_input[:, 1:] @ torch.linalg.pinv(mass_positive_input[:, 1:] + mass_positive_input[:, :-1])
            mass_scale2_pos = mass_positive_input[:, :-1] @ torch.linalg.pinv(mass_positive_input[:, 1:] + mass_positive_input[:, :-1])
            mass_scale_pos = torch.cat((mass_scale1_pos, -mass_scale2_pos), dim=1).view(-1, self.n_edge, 3, 3)

            undeform_vert_positive = b_undeformed_vert_pre_constraint+d_positions_init
            undeform_vert_positive_input = undeform_vert_positive.clone()
            undeform_vert_negative = b_undeformed_vert_pre_constraint-d_positions_init
            undeform_vert_negative_input = undeform_vert_negative.clone()


            positions_ICE_neg, _ = self.constraints_enforcement.Inextensibility_Constraint_Enforcement(
                self.batch,
                positions_negative_input,
                self.batched_m_restEdgeL_neg,  # change to nominal length
                mass_negative_input,
                self.clamped_index,
                self.inext_scale,
                mass_scale_neg,
                self.zero_mask_num,
                undeform_vert_negative_input,
                self.bkgrad_neg,
                self.n_branch
            )


            delta_positions_ICE_neg = (positions_ICE_neg - positions_negative)
            # print('delta_positions_ICE_neg', delta_positions_ICE_neg)

            positions_ICE_pos, _ = self.constraints_enforcement.Inextensibility_Constraint_Enforcement(
                self.batch,
                positions_positive_input,
                self.batched_m_restEdgeL_pos,  # change to nominal length
                mass_positive_input,
                self.clamped_index,
                self.inext_scale,
                mass_scale_pos,
                self.zero_mask_num,
                undeform_vert_positive_input,
                self.bkgrad_pos,
                self.n_branch
            )


            delta_positions_ICE_pos = (positions_ICE_pos - positions_positive)


            d_delta_positions_ICE = (delta_positions_ICE_pos - delta_positions_ICE_neg) / 2
            d_delta_positions_ICE_np = d_delta_positions_ICE.detach().cpu().numpy()






            #mass_matrix expand to (1,12,12)diagonally position flatten (1,12,1)
            print('numerical', d_delta_positions_ICE_np)
            print('analytical',analytical_d_delta_positions_ICE)
            print('multiple in iter_Sim',analytical_d_delta_positions_ICE[0]*1/d_delta_positions_ICE_np[0])


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
