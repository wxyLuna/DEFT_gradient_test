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
import gradients
import numpy as np
from scipy.optimize import check_grad
import re



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
                                               [0.0, 0.0, 0.0],
                                               [1.0, 3.0, 5.0],
                                               [0.0, 1.0, 4.0],
                                               [2.0, 4.0, 6.0],
                                               [0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.0]]]) * 1e-6
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
        self.bkgrad = gradients.BackwardGradientIC(self.batch *n_branch, n_vert)
        self.bkgrad_neg = gradients.BackwardGradientIC(self.batch * n_branch, n_vert)
        self.bkgrad_pos = gradients.BackwardGradientIC(self.batch * n_branch, n_vert)

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
        self.clamped_index = torch.tensor([[1.0, 0.0,0,0,0, 1.0, 1.0]]) # hardcoded clamped index for the first vertex
        self.inext_scale = self.clamped_index * (1e20)+1 # clamped points does not move
        self.n_branch=n_branch
        # self.damping = nn.Parameter(torch.tensor(0.1, device=device))  # damping factor
        self.parent_clamped_selection = torch.tensor((0, 1, -2, -1))



    def External_Force(self, mass_matrix
                       ):
        return torch.matmul(mass_matrix, self.gravity.view(-1, 1)).squeeze(-1)

    def Numerical_Integration(self,mass_matrix,total_force, velocities,positions, dt):
        acc = torch.linalg.solve(mass_matrix, total_force.unsqueeze(-1)).squeeze(-1)
        velocities = velocities + acc * dt
        positions = positions + velocities * dt  # not wrapped in nn.Parameter
        return positions


    def save_and_later_average_errors(self,ratio, relative_error, absolute_error, timer, time_step, save_dir="error_logs",
                                      mode="save"):
        """
        Save or load + average error arrays (ratio, relative, absolute) for a given timer and time_step.

        Args:
            ratio (np.ndarray): The ratio array (batch, n_vert, 3).
            relative_error (np.ndarray): Relative error array.
            absolute_error (np.ndarray): Absolute error array.
            timer (int): Current outer loop counter.
            time_step (int): Current inner time step.
            save_dir (str): Directory to save or load data.
            mode (str): Either "save" to write files or "load" to compute means from saved data.
        """
        os.makedirs(save_dir, exist_ok=True)

        def find_next_log_index():
            existing = [d for d in os.listdir(save_dir) if re.match(r"^error_log\d+$", d)]
            indices = [int(d.replace("error_log", "")) for d in existing]
            return max(indices) + 1 if indices else 1

        if not hasattr(self, "current_error_log_dir"):
            log_id = find_next_log_index()
            self.current_error_log_dir = os.path.join(save_dir, f"error_log{log_id}")
            os.makedirs(self.current_error_log_dir, exist_ok=True)
            print(f"[INFO] Writing to new log folder: {self.current_error_log_dir}")


        if mode == "save":
            # Save each array with unique names
            np.save(os.path.join(self.current_error_log_dir, f"ratio_timer{timer}_step{time_step}.npy"), ratio)
            np.save(os.path.join(self.current_error_log_dir, f"relerr_timer{timer}_step{time_step}.npy"), relative_error)
            np.save(os.path.join(self.current_error_log_dir, f"abserr_timer{timer}_step{time_step}.npy"), absolute_error)

        elif mode == "load":
            # Load all matching files and compute averaged values
            ratio_vals, rel_vals, abs_vals = [], [], []

            for filename in os.listdir(save_dir):
                full_path = os.path.join(save_dir, filename)
                if filename.startswith("ratio_") and filename.endswith(".npy"):
                    ratio_vals.append(np.load(full_path))
                elif filename.startswith("relerr_") and filename.endswith(".npy"):
                    rel_vals.append(np.load(full_path))
                elif filename.startswith("abserr_") and filename.endswith(".npy"):
                    abs_vals.append(np.load(full_path))

            if ratio_vals:
                avg_ratio = np.mean(np.concatenate(ratio_vals)) / 10
                print(f"Averaged Ratio / 10: {avg_ratio:.3e}")
            if rel_vals:
                avg_rel_error = np.mean(np.concatenate(rel_vals)) / 10
                print(f"Averaged Relative Error / 10: {avg_rel_error:.3e}")
            if abs_vals:
                avg_abs_error = np.mean(np.concatenate(abs_vals)) / 10
                print(f"Averaged Absolute Error / 10: {avg_abs_error:.3e}")

    def iterative_sim(self, time_horizon, positions_traj, previous_positions_traj,target_traj, loss_func,dt,timer):
        traj_loss_eval = 0.0
        total_loss = 0.0
        total_force = self.External_Force(self.mass_matrix)
        constraint_loop = 20
        # Enforce clamp constraints

        # parent_fix_point = self.undeformed_vert[:, :, 0, self.parent_clamped_selection]
        # positions_traj[:, :, 0, self.parent_clamped_selection] = parent_fix_point
        # previous_positions_traj[:, :, 0, self.parent_clamped_selection] = parent_fix_point
        # target_traj[:, :, 0, self.parent_clamped_selection] = parent_fix_point



        for t in range(int(time_horizon)):

            self.bkgrad.reset(self.batch, self.n_vert)

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



            # ___Analytical gradient & Center values for inextensibility constraint enforcement___
            for _ in range(constraint_loop):
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
            # print('self.bkgrad.grad_DX_X', self.bkgrad.grad_DX_X)

            # ___Analytical perturbation___

            # ___Continue with the simulation using the enforced positions___
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

    def constraint_loop_iteration(self, constraint_loop, batch, positions, nominal_length, mass_matrix, inext_scale, clamped_index,
                            mass_scale, zero_mask_num, b_undeformed_vert, bkgrad, n_branch):
        '''Iterative simulation loop for constraint satisfaction.'''

        for _ in range(constraint_loop):
            positions_ICE, grad_per_ICitr = self.constraints_enforcement.Inextensibility_Constraint_Enforcement(
                batch,
                positions,
                nominal_length,  ## change to nominal length
                mass_matrix,
                clamped_index,
                inext_scale,
                mass_scale,
                zero_mask_num,
                b_undeformed_vert,
                bkgrad,
                n_branch
            )
            bkgrad.grad_DX_X = grad_per_ICitr.grad_DX_X
            bkgrad.grad_DX_Xinit = grad_per_ICitr.grad_DX_Xinit
            bkgrad.grad_DX_M = grad_per_ICitr.grad_DX_M

        return positions_ICE

    def generate_preX_trajectory(self, time_horizon, dt):
        # Initialize tensors
        b_DLOs_vertices_traj = torch.zeros(time_horizon, self.batch * self.n_branch, self.n_vert, 3)

        # Initial positions and velocities
        positions_t = self.undeformed_vert.clone().detach()  # shape: [batch, n_vert, 3]
        velocities_t = torch.zeros_like(positions_t)

        for t in range(time_horizon):
            # Step 1–5 in Algorithm 1:
            total_force = self.External_Force(self.mass_matrix)  # Apply gravity
            positions_t1 = self.Numerical_Integration(self.mass_matrix, total_force, velocities_t,
                                                   positions_t, dt)
            positions_t1_before_clamp = positions_t1.clone()
            positions_t1[:,self.parent_clamped_selection,:] = self.undeformed_vert[:,self.parent_clamped_selection,:].detach() # Apply clamped constraints
            print('clamp',positions_t1-positions_t1_before_clamp)
            # Enforce inextensibility constraint (Step 4)
            for _ in range(1):  # constraint_loop
                positions_t1, _ = self.constraints_enforcement.predX_Inextensibility_Constraint_Enforcement(
                    self.batch,
                    positions_t1,
                    self.batched_m_restEdgeL,
                    self.mass_matrix,
                    self.clamped_index,
                    self.inext_scale,
                    self.mass_scale,
                    self.zero_mask_num,
                    self.b_undeformed_vert,
                    self.bkgrad,
                    self.n_branch
                )

            # Update velocity after constraint enforcement
            velocities_t = (positions_t1 - positions_t) / dt

            # Save trajectory
            b_DLOs_vertices_traj[t] = positions_t1.detach().cpu()

            # Prepare for next step
            positions_t = positions_t1.clone()

        return b_DLOs_vertices_traj  # shape: [time_horizon, batch * n_branch, n_vert, 3]
