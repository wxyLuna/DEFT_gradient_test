import torch
import torch.nn as nn
from numpy.ma.core import absolute

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
from util import rotation_matrix, computeW, computeLengths, computeEdges
import gradients
import numpy as np
import re




class Unit_test_sim(nn.Module):
    def __init__(self, batch,
                        n_vert,
                        n_branch,
                        cs_n_vert,
                        n_edge,
                        b_DLO_mass,
                        b_undeformed_vert,
                        parent_DLO_MOI,
                        children_DLO_MOI,
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
                        device):
        super().__init__()
        self.n_vert = n_vert
        self.n_edge = n_edge
        self.device = device
        self.batch = batch
        self.n_branch = n_branch
        self.clamp_parent = clamp_parent
        self.clamp_child1 = clamp_child1
        self.clamp_child2 = clamp_child2
        # For identifying the branches in a batch:
        # - child1 is 1 mod n_branch
        # - child2 is 2 mod n_branch
        # - parent is 0 mod n_branch
        selected_child1_index = list(range(1, batch * n_branch, n_branch))
        selected_child2_index = list(range(2, batch * n_branch, n_branch))
        selected_parent_index = list(range(0, batch * n_branch, n_branch))
        self.selected_parent_index = torch.tensor(selected_parent_index)
        self.selected_child1_index = torch.tensor(selected_child1_index)
        self.selected_child2_index = torch.tensor(selected_child2_index)
        # For parallelization across a batch and multiple branches:
        # We'll figure out child vs parent branches (indices) in a vectorized way
        selected_children_index = [i for i in range(1, batch * n_branch) if i % n_branch != 0]
        self.selected_children_index = selected_children_index

        # Expand clamped vertex selection for the parent across the batch
        batch_indices = self.selected_parent_index.unsqueeze(1).expand(-1, parent_clamped_selection.size(0))
        parent_indices = parent_clamped_selection.unsqueeze(0).expand(self.selected_parent_index.size(0), -1)

        # Child1/Child2 clamped indices
        batch_child1_indices = self.selected_child1_index
        child1_indices = child1_clamped_selection
        batch_child2_indices = self.selected_child2_index
        child2_indices = child2_clamped_selection
        # Flatten them for easier indexing
        self.batch_indices_flat = batch_indices.reshape(-1)
        self.parent_indices_flat = parent_indices.reshape(-1)

        self.batch_child1_indices_flat = batch_child1_indices.reshape(-1)
        self.child1_indices_flat = child1_indices.reshape(-1)

        self.batch_child2_indices_flat = batch_child2_indices.reshape(-1)
        self.child1_indices_flat = child2_indices.reshape(-1)  # reusing variable name but it's for child2

        self.b_undeformed_vert = b_undeformed_vert.clone()
        self.zero_mask = torch.all(self.b_undeformed_vert[:, 1:] == 0, dim=-1)
        self.zero_mask_num = 1 - self.zero_mask.repeat(batch, 1).to(torch.uint8)

        self.d_positions_init = torch.zeros_like(self.b_undeformed_vert) #initialize the initial positions perturbation to zero
        self.m_restEdgeL, self.m_restRegionL = computeLengths(
            computeEdges(self.b_undeformed_vert.clone(), self.zero_mask)
        )
        # Create a mask to handle situations where child branches end sooner
        m_restRegionL_mask = torch.ones_like(self.m_restRegionL)
        for i in range(len(cs_n_vert)):
            m_restRegionL_mask[i + 1, cs_n_vert[i] - 1:] = 0.
        # Apply the masks so that unused edges are 0
        self.m_restRegionL = self.m_restRegionL * m_restRegionL_mask
        self.m_restEdgeL = self.m_restEdgeL * m_restRegionL_mask
        self.batched_m_restEdgeL = self.m_restEdgeL.repeat(self.batch, 1, 1).view(-1, n_edge)

        self.m_restEdgeL_pos, self.m_restRegionL_pos = computeLengths(
            computeEdges(self.b_undeformed_vert.clone()+self.d_positions_init.clone(), self.zero_mask)
        )
        self.batched_m_restEdgeL_pos = self.m_restEdgeL_pos.repeat(self.batch, 1, 1).view(-1, n_edge)

        self.m_restEdgeL_neg, self.m_restRegionL_neg = computeLengths(
            computeEdges(self.b_undeformed_vert.clone() - self.d_positions_init.clone(), self.zero_mask)
        )
        self.batched_m_restEdgeL_neg = self.m_restEdgeL_neg.repeat(self.batch, 1, 1).view(-1, n_edge)
        self.undeformed_vert = nn.Parameter(self.b_undeformed_vert)
        ## for storing the old gradients from inextensibility enforcement
        self.bkgrad = gradients.BackwardGradientIC(self.batch *n_branch, n_vert)
        self.bkgrad_neg = gradients.BackwardGradientIC(self.batch * n_branch, n_vert)
        self.bkgrad_pos = gradients.BackwardGradientIC(self.batch * n_branch, n_vert)
        ## for storing the old gradients from Numerical integration
        self.bkgrad_damping = gradients.BackwardGradientDamping(self.batch, n_branch, n_vert)
        self.bkgrad_IR = gradients.BackwardGradientIR(self.batch*n_branch, n_vert)
        # for storing old gradients from ICE coupling constraints
        self.bkgrad_coupling = gradients.BackwardGradientCoupling(self.batch*n_branch, n_vert)


        self.gravity = nn.Parameter(torch.tensor((0, 0, -9.81), device=device))
        self.dt = 1e-2

        self.clamped_index = clamped_index

        self.mass_diagonal = nn.Parameter(b_DLO_mass)
        self.mass_matrix = (
                torch.eye(3)
                .unsqueeze(dim=0)
                .unsqueeze(dim=0)
                .repeat(n_branch, n_vert, 1, 1)
                * (self.mass_diagonal.unsqueeze(dim=-1).unsqueeze(dim=-1))
        ).unsqueeze(dim=0).repeat(batch, 1, 1, 1, 1).view(-1, n_vert, 3, 3)
        mass_scale1 = self.mass_matrix[:, 1:] @ torch.linalg.pinv(self.mass_matrix[:, 1:] + self.mass_matrix[:, :-1])
        mass_scale2 = self.mass_matrix[:, :-1] @ torch.linalg.pinv(self.mass_matrix[:, 1:] + self.mass_matrix[:, :-1])
        self.mass_scale = torch.cat((mass_scale1, -mass_scale2), dim=1).view(-1, self.n_edge, 3, 3)
        self.constraints_enforcement = constraints_enforcement(n_branch)
        self.parent_clamped_selection = torch.tensor((0, 1, -2, -1),device=device)  # hardcoded parent clamped selection
        self.child1_clamped_selection = torch.tensor((0), device=device)  # hardcoded child1 clamped selection
        self.child2_clamped_selection = torch.tensor((0), device=device)

        inext_scale = self.clamped_index * 1e20
        self.inext_scale = (inext_scale + 1.).repeat(batch, 1)
        self.inext_scale = torch.cat((self.inext_scale[:, :-1], self.inext_scale[:, 1:]), dim=1).view(-1, n_edge)
        self.n_branch = n_branch
        self.damping = damping
        self.d_damping = torch.tensor(0*1e-6, device=device)
        self.damping_pos = nn.Parameter(self.damping + self.d_damping)
        self.damping_neg = nn.Parameter(self.damping - self.d_damping)

        self.integration_ratio = nn.Parameter(torch.tensor(1., device=device))
        self.d_integration_ratio = torch.tensor(5*1e-6, device=device)
        self.integration_ratio_pos = nn.Parameter(self.integration_ratio + self.d_integration_ratio)
        self.integration_ratio_neg = nn.Parameter(self.integration_ratio - self.d_integration_ratio)

        self.rigid_body_coupling_index = rigid_body_coupling_index
        # Store inertia (MOI) for parent/child in parameter form
        self.p_DLO_diagonal = nn.Parameter(parent_DLO_MOI)
        self.c_DLO_diagonal = nn.Parameter(children_DLO_MOI)

        # Construct MOI matrices for children/parent rods
        self.children_MOI_matrix = torch.zeros(n_branch - 1, 3, 3)
        self.children_MOI_matrix[:, 0, 0] = self.c_DLO_diagonal[:, 0]
        self.children_MOI_matrix[:, 1, 1] = self.c_DLO_diagonal[:, 1]
        self.children_MOI_matrix[:, 2, 2] = self.c_DLO_diagonal[:, 2]

        self.parent_MOI_matrix = torch.zeros((n_branch - 1) * 2, 3, 3)
        self.parent_MOI_matrix[:, 0, 0] = self.p_DLO_diagonal[:, 0]
        self.parent_MOI_matrix[:, 1, 1] = self.p_DLO_diagonal[:, 1]
        self.parent_MOI_matrix[:, 2, 2] = self.p_DLO_diagonal[:, 2]

        # We compute momentum scaling factors for rotation constraints
        self.parent_MOI_index1 = parent_MOI_index1
        self.parent_MOI_index2 = parent_MOI_index2

        self.parent_MOI_index1 = parent_MOI_index1
        self.parent_MOI_index2 = parent_MOI_index2

        rod_MOI1, rod_MOI2 = self.parent_MOI_matrix[parent_MOI_index1].repeat(batch, 1,1), self.children_MOI_matrix.repeat(batch,1, 1)
        momentum_scale1 = -rod_MOI2 @ torch.linalg.pinv(rod_MOI1 + rod_MOI2)
        momentum_scale2 = rod_MOI1 @ torch.linalg.pinv(rod_MOI1 + rod_MOI2)
        self.momentum_scale_previous = torch.cat((momentum_scale1, momentum_scale2), dim=1).view(-1, 3, 3)

        rod_MOI1, rod_MOI2 = self.parent_MOI_matrix[parent_MOI_index2].repeat(batch, 1,1), self.children_MOI_matrix.repeat(batch,1, 1)
        momentum_scale1 = -rod_MOI2 @ torch.linalg.pinv(rod_MOI1 + rod_MOI2)
        momentum_scale2 = rod_MOI1 @ torch.linalg.pinv(rod_MOI1 + rod_MOI2)
        self.momentum_scale_next = torch.cat((momentum_scale1, momentum_scale2), dim=1).view(-1, 3, 3)

        # Next, we compute coupling mass scale for the branching points
        parent_mass = self.mass_matrix[selected_parent_index][:, rigid_body_coupling_index].view(-1, 3, 3)
        children_mass = self.mass_matrix[selected_children_index, 0]
        self.selected_children_index = selected_children_index
        mass_scale1 = children_mass @ torch.linalg.inv(parent_mass + children_mass)
        mass_scale2 = parent_mass @ torch.linalg.inv(parent_mass + children_mass)
        self.coupling_mass_scale = torch.cat((mass_scale1.unsqueeze(dim=1), -mass_scale2.unsqueeze(dim=1)), dim=1)
        self.parent_mass = parent_mass
        self.children_mass = children_mass

    def External_Force(self, mass_matrix
                       ):
        return torch.matmul(mass_matrix, self.gravity.view(-1, 1)).squeeze(-1)

    def Numerical_Integration(self,mass_matrix,total_force, velocities,positions, damping, integration_ratio, dt):
        '''Perform numerical integration using the mass matrix and total force.'''

        # update gradient for corresponding vertices
        grad_DX_damping = gradients.grad_DX_damping_batch(self.n_vert, integration_ratio, dt, velocities, self.n_branch)
        grad_DX_IR = gradients.grad_DX_IR_batch(dt, velocities, mass_matrix, total_force, damping)

        velocities = velocities.clone() + (
                (
                        total_force.unsqueeze(dim=-2)
                        - velocities.unsqueeze(dim=-2) * damping.repeat(self.batch).clone().view(-1, 1, 1, 1)
                        * self.mass_diagonal.repeat(self.batch, 1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                )@ torch.linalg.pinv(mass_matrix) * dt).reshape(-1, velocities.size()[1], 3)

        #
        positions = positions.clone() + velocities * dt * integration_ratio

        return positions, grad_DX_damping, grad_DX_IR


    def save_and_later_average_errors(self,ratio, relative_error, absolute_error, timer, time_step, save_dir,
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
    def Rod_init(self, batch, m_restEdgeL, clamped_index):
        """
        Initialize the rod with undeformed vertices and compute rest lengths.
        """
        undeformed_vert, _ = self.constraints_enforcement.Inextensibility_Constraint_Enforcement(
                    batch,
                    (self.undeformed_vert.clone()).repeat(batch, 1, 1),
                    m_restEdgeL, ## change to nominal length
                    self.mass_matrix,
                    clamped_index,
                    self.inext_scale,
                    self.mass_scale,
                    self.zero_mask_num,
                    self.b_undeformed_vert,
                    self.bkgrad,
                    self.n_branch
                )
        return undeformed_vert


    def iterative_sim(self, time_horizon, positions_traj, previous_positions_traj,target_traj, loss_func,dt,timer):
        traj_loss_eval = 0.0
        total_loss = 0.0
        total_force = self.External_Force(self.mass_matrix)
        constraint_loop = 20





        for t in range(int(time_horizon)):

            self.bkgrad.reset(self.batch,self.n_vert)
            self.bkgrad_damping.reset(self.batch,self.n_branch, self.n_vert)
            self.bkgrad_IR.reset(self.batch, self.n_vert)

            if t == 0:
                # print('at time step', t)
                positions = positions_traj[:,t].reshape(-1, self.n_vert, 3)
                prev_positions = previous_positions_traj[:,t].reshape(-1, self.n_vert, 3)
                velocities = (positions - prev_positions) / dt
            else:
                # print('else at time step', t)

                prev_positions = positions_old.clone()

            positions_input = positions.clone()
            positions, bkgrad_damping, bkgrad_IR = self.Numerical_Integration(self.mass_matrix, total_force, velocities,
                                                               positions_input, self.damping, self.integration_ratio, dt)
            self.bkgrad_damping.grad_DX_damping = bkgrad_damping
            self.bkgrad_IR.grad_DX_IR = bkgrad_IR
            if self.clamp_parent:
                parent_fix_point = target_traj[:, :, 0, self.parent_clamped_selection]
                parent_fix = parent_fix_point[:, t].reshape(-1, 3)
                positions[self.batch_indices_flat, self.parent_indices_flat] = parent_fix

            if self.clamp_child1:
                child1_fix_point = target_traj[:, :, 1, self.child1_clamped_selection]
                c1_fix = child1_fix_point[:, t].reshape(-1, 3)
                positions[self.batch_child1_indices_flat, self.child1_indices_flat] = c1_fix

            if self.clamp_child2:
                child2_fix_point = target_traj[:, :, 2, self.child2_clamped_selection]
                c2_fix = child2_fix_point[:, t].reshape(-1, 3)
                positions[self.batch_child2_indices_flat, self.child2_indices_flat] = c2_fix


            # ___Analytical gradient & Center values for inextensibility constraint enforcement___

            for _ in range(constraint_loop):
                parent_vertices = positions[self.selected_parent_index]
                children_vertices = positions[self.selected_children_index].view(self.batch, -1, self.n_vert, 3)

                children_vertices = children_vertices.view(-1, self.n_vert, 3)
                #coupling constraints
                positions, grad_per_Coupling_itr = self.constraints_enforcement.Inextensibility_Constraint_Enforcement_Coupling(
                    parent_vertices,
                    children_vertices,
                    self.rigid_body_coupling_index,
                    self.coupling_mass_scale,
                    self.parent_mass,
                    self.children_mass,
                    self.selected_parent_index,
                    self.selected_children_index,
                    self.bkgrad_coupling
                )


                #Inextensibility constraint
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
                # self.bkgrad.grad_DX_Xinit = grad_per_ICitr.grad_DX_Xinit
                self.bkgrad.grad_DX_M = grad_per_ICitr.grad_DX_M



            # ___Continue with the simulation using the enforced positions___
            velocities = (positions_ICE - prev_positions) / dt

            gt_positions = target_traj[:, t].reshape(-1, self.n_vert, 3)
            gt_velocities = (target_traj[:, t] - positions_traj[:, t]).view(-1,self.n_vert,3) / dt
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
            # bkgrad.grad_DX_X = grad_per_ICitr.grad_DX_X
            # bkgrad.grad_DX_Xinit = grad_per_ICitr.grad_DX_Xinit
            # bkgrad.grad_DX_M = grad_per_ICitr.grad_DX_M

        return positions_ICE

    def generate_preX_trajectory(self, time_horizon, dt):
        # Initialize tensors
        b_DLOs_vertices_traj = torch.zeros(time_horizon, self.batch * self.n_branch, self.n_vert, 3)
        parent_rod_axis_angle = torch.zeros(1, 3)
        parent_rod_orientation = pytorch3d.transforms.rotation_conversions.axis_angle_to_quaternion(
            parent_rod_axis_angle
        ).unsqueeze(dim=0).repeat(self.batch, self.n_vert - 1, 1)
        child_rod_axis_angle = torch.zeros(1, 3)
        children_rod_orientation = pytorch3d.transforms.rotation_conversions.axis_angle_to_quaternion(
            child_rod_axis_angle
        ).unsqueeze(dim=0).repeat(self.batch, len(self.rigid_body_coupling_index), 1)
        # For parent-child constraints iteration
        previous_parent_vertices_iteration_edge1 = None
        previous_parent_vertices_iteration_edge2 = None
        previous_children_vertices_iteration_edge = None


        # Initial positions and velocities
        positions_t = self.undeformed_vert.clone().detach()  # shape: [batch, n_vert, 3]
        velocities_t = torch.zeros_like(positions_t)
        positions_t[:, self.parent_clamped_selection, :] = self.undeformed_vert[:, self.parent_clamped_selection,:].detach()
        previous_parent_vertices_iteration_edge1 = positions_t[self.selected_parent_index].clone()
        previous_parent_vertices_iteration_edge2 = positions_t[self.selected_parent_index].clone()
        previous_children_vertices_iteration_edge = positions_t[self.selected_children_index].view(self.batch, -1,self.n_vert,3).clone()

        for t in range(time_horizon):
            # Step 1–5 in Algorithm 1:

            total_force = self.External_Force(self.mass_matrix)  # Apply gravity
            positions_t1,_, _ = self.Numerical_Integration(self.mass_matrix, total_force, velocities_t,
                                                   positions_t, self.damping, self.integration_ratio, dt)
            positions_t1_clamp_selection = positions_t1.clone()
            # positions_t1_clamp_selection[:, self.parent_clamped_selection, :] = self.undeformed_vert[:, self.parent_clamped_selection,:].detach()
            if self.clamp_parent:
                parent_fix_point = self.undeformed_vert[0, self.parent_clamped_selection]
                positions_t1_clamp_selection[ 0, self.parent_clamped_selection] = parent_fix_point

            if self.clamp_child1:
                child1_fix_point = self.undeformed_vert[ 1, self.child1_clamped_selection]
                positions_t1_clamp_selection[1, self.child1_clamped_selection] = child1_fix_point

            if self.clamp_child2:
                child2_fix_point = self.undeformed_vert[2, self.child2_clamped_selection]
                positions_t1_clamp_selection[2, self.child2_clamped_selection] = child2_fix_point
            # Enforce inextensibility constraint (Step 4)
            for _ in range(10):  # constraint_loop
                parent_vertices = positions_t1_clamp_selection[self.selected_parent_index]
                children_vertices = positions_t1_clamp_selection[self.selected_children_index].view(self.batch, -1, self.n_vert, 3)

                # # Edge1
                # parent_vertices, parent_rod_orientation, children_vertices, children_rod_orientation = \
                #     self.constraints_enforcement.Rotation_Constraints_Enforcement_Parent_Children(
                #         parent_vertices,
                #         parent_rod_orientation,
                #         previous_parent_vertices_iteration_edge1,
                #         children_vertices,
                #         children_rod_orientation,
                #         previous_children_vertices_iteration_edge,
                #         self.parent_MOI_matrix,
                #         self.children_MOI_matrix,
                #         torch.tensor(self.rigid_body_coupling_index) - 1,
                #         torch.linspace(0, (children_vertices.size(1) * 2 - 2), len(self.rigid_body_coupling_index)).to(
                #             torch.int),
                #         self.momentum_scale_previous
                #     )
                #
                # previous_parent_vertices_iteration_edge1 = parent_vertices.clone()
                # previous_children_vertices_iteration_edge = children_vertices.clone()
                #
                # # Edge2
                # parent_vertices, parent_rod_orientation, children_vertices, children_rod_orientation = \
                #     self.constraints_enforcement.Rotation_Constraints_Enforcement_Parent_Children(
                #         parent_vertices,
                #         parent_rod_orientation,
                #         previous_parent_vertices_iteration_edge2,
                #         children_vertices,
                #         children_rod_orientation,
                #         previous_children_vertices_iteration_edge,
                #         self.parent_MOI_matrix,
                #         self.children_MOI_matrix,
                #         torch.tensor(self.rigid_body_coupling_index),
                #         torch.linspace(1, (children_vertices.size(1) * 2 - 1), len(self.rigid_body_coupling_index)).to(
                #             torch.int),
                #         self.momentum_scale_next
                #     )
                # previous_parent_vertices_iteration_edge2 = parent_vertices.clone()
                # previous_children_vertices_iteration_edge = children_vertices.clone()






                children_vertices = children_vertices.view(-1, self.n_vert, 3)
                # coupling constraints
                positions,_ = self.constraints_enforcement.Inextensibility_Constraint_Enforcement_Coupling(
                    parent_vertices,
                    children_vertices,
                    self.rigid_body_coupling_index,
                    self.coupling_mass_scale,
                    self.parent_mass,
                    self.children_mass,
                    self.selected_parent_index,
                    self.selected_children_index,
                    self.bkgrad_coupling
                )

                positions_ICE, _ = self.constraints_enforcement.Inextensibility_Constraint_Enforcement(
                    self.batch,
                    positions,
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
            velocities_t = (positions_ICE - positions_t) / dt
            # Save trajectory
            b_DLOs_vertices_traj[t] = positions_ICE.detach().cpu()

            # Prepare for next step
            positions_t = positions_ICE.clone()

        return b_DLOs_vertices_traj  # shape: [time_horizon, batch * n_branch, n_vert, 3]
