import torch
import numpy as np
from constraints_solver import constraints_enforcement
from Unit_test_sim import Unit_test_sim
##this function is used to test the outer loop gradient of the iterative simulation
def constraint_loop_sim(constraint_loop,batch, positions,nominal_length,mass_matrix,inext_scale,clamped_index,mass_scale,
            zero_mask_num,b_undeformed_vert,bkgrad,n_branch):
    '''Iterative simulation loop for constraint satisfaction.'''

    for _ in range(constraint_loop):
        positions_ICE, grad_per_ICitr = constraints_enforcement.Inextensibility_Constraint_Enforcement(
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