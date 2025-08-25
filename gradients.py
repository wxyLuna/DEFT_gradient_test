import numpy as np
import torch

# Gradient Saver
class BackwardGradientIC:
    def __init__(self, batch, num_branch,num_vertices):
        # the batch here is actually num_batch * num_branch
        self.batch = batch
        self.num_branch = num_branch
        self.num_vertices = num_vertices
        self.grad_DX_X = None
        self.grad_DX_Xinit = None
        self.grad_DX_M = None
        self.reset(batch, num_branch, num_vertices)

        return

    def reset(self, batch,num_branch, num_vertices):
        self.grad_DX_X = np.zeros((batch, num_branch*num_vertices*3, num_branch*num_vertices*3), dtype=np.float32) ## change dimension
        # self.grad_DX_Xinit = np.zeros((batch, num_branch*num_vertices*3, num_branch*num_vertices*3), dtype=np.float32)
        self.grad_DX_M = np.zeros((batch, num_branch*num_vertices*3, num_branch*num_vertices), dtype=np.float32)## change dimension
        return

class BackwardGradientDamping:
    def __init__(self, batch, branch, num_vertices):
        # the batch here is actually num_batch * num_branch, while the branch is num_branch
        self.grad_DX_damping = None
        self.reset(batch, branch, num_vertices)
        return

    def reset(self, batch, branch, num_vertices):
        self.grad_DX_damping = np.zeros((batch, num_vertices*3, branch), dtype=np.float32)
        return

class BackwardGradientIR:
    def __init__(self, batch, num_vertices):
        # the batch here is actually num_batch * num_branch
        self.grad_DX_IR = None
        self.reset(batch, num_vertices)
        return

    def reset(self, batch, num_vertices):
        self.grad_DX_IR = np.zeros((batch, num_vertices*3, 1), dtype=np.float32)
        return

class BackwardGradientCoupling:
    def __init__(self, batch, num_vertices):
        # the batch here is actually num_batch * num_branch
        self.grad_DX_M_Coupling = None
        self.grad_DX_X_Coupling = None
        self.reset(batch, num_vertices)
        return

    def reset(self, batch, num_vertices):
        self.grad_DX_M_Coupling = np.zeros((batch, 2*3, 2), dtype=np.float32)
        self.grad_DX_X_Coupling = np.zeros((batch, 2*3, 6), dtype=np.float32)
        return

# Gradient Solver
#Inextensibility Constraint Enforcement

def grad_DX_X_ICitr_batch(M_0, M_1, X_0, X_1, X_0_init, X_1_init, mask):
    """
    Robust batch gradient for inextensibility wrt positions.
    Inputs:
      M_0, M_1:        (B,3,3)
      X_0, X_1:        (B,3,1)
      X_0_init, X_1_init: (B,3,1)
      mask:            (B,) or (B,1) or (B,1,1); True = active edge
    Output:
      grad_DX_X:       (B,6,6)
    """
    eps = 1e-12

    # ---- to numpy (safe if torch tensors) ----
    def to_np(x): return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    M_0, M_1 = to_np(M_0), to_np(M_1)
    X_0, X_1 = to_np(X_0), to_np(X_1)
    X_0_init, X_1_init = to_np(X_0_init), to_np(X_1_init)
    B = M_0.shape[0]

    # ---- normalize mask to (B,1,1) ----
    mask = to_np(mask).astype(bool)
    if mask.ndim == 1:
        mask = mask[:, None, None]
    elif mask.ndim == 2:
        mask = mask[:, :, None]

    # ---- detect inactive/invalid batches ----
    m0_zero = (np.abs(M_0).sum(axis=(1,2), keepdims=True) == 0)
    m1_zero = (np.abs(M_1).sum(axis=(1,2), keepdims=True) == 0)
    x1_zero = (np.abs(X_1).sum(axis=(1,2), keepdims=True) == 0)
    x1i_zero= (np.abs(X_1_init).sum(axis=(1,2), keepdims=True) == 0)

    bothM_zero = m0_zero & m1_zero                  # cannot invert
    tail_zero  = m1_zero & x1_zero & x1i_zero       # tail vertex absent
    active     = (~bothM_zero) & (~tail_zero) & mask   # (B,1,1)

    # ---- default output zeros ----
    grad_DX_X = np.zeros((B, 6, 6), dtype=np.float64)
    if not np.any(active):
        return grad_DX_X

    idx = np.where(active.reshape(B))[0]
    M0v, M1v = M_0[idx], M_1[idx]
    X0v, X1v = X_0[idx], X_1[idx]
    X0iv, X1iv = X_0_init[idx], X_1_init[idx]
    b = len(idx)

    # ---- edge geometry ----
    Edge      = X1v - X0v           # (b,3,1)
    Edge_init = X1iv - X0iv         # (b,3,1)
    L  = np.linalg.norm(Edge,      axis=1, keepdims=True)   # (b,1,1)
    L0 = np.linalg.norm(Edge_init, axis=1, keepdims=True)   # (b,1,1)

    denom = L**2 + L0**2
    denom = np.where(denom < eps, eps, denom)

    # your scalars
    lam   = (L**2 - L0**2) / denom                 # (b,1,1)
    scale = 4.0 * (L0**2) / (denom**2)             # (b,1,1)

    # ---- safe inverse of (M0+M1) ----
    sumM    = M0v + M1v                            # (b,3,3)
    M_param = np.linalg.pinv(sumM, rcond=1e-12)    # (b,3,3) robust
    # outer product Edge*Edge^T: (b,3,3)
    E_outer = np.einsum('bik,bjk->bij', Edge, Edge)

    # Blocks (b,3,3), matching your formulas
    # Note: M_param already acts like your previous M_param;
    # scaled_value == scale; lambda_param == lam.

    term_M1  = M1v @ M_param                       # (b,3,3)
    term_M0  = M0v @ M_param                       # (b,3,3)
    M1_MP_EE = np.einsum('bij,bjk,bkl->bil', M1v, M_param, E_outer)
    M0_MP_EE = np.einsum('bij,bjk,bkl->bil', M0v, M_param, E_outer)

    grad_00 = -M1_MP_EE * scale - term_M1 * lam
    grad_01 =  M1_MP_EE * scale + term_M1 * lam
    grad_10 =  M0_MP_EE * scale + term_M0 * lam
    grad_11 = -M0_MP_EE * scale - term_M0 * lam

    # pack (b,6,6)
    top = np.concatenate([grad_00, grad_01], axis=2)  # (b,3,6)
    bot = np.concatenate([grad_10, grad_11], axis=2)  # (b,3,6)
    packed = np.concatenate([top, bot], axis=1)       # (b,6,6)

    grad_DX_X[idx] = packed
    return grad_DX_X


def grad_DX_Xinit_ICitr_batch(M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Batch version of Gradient of the inextensibility constraint iterative function with respect to the undeformed positions X_0_init and X_1_init.

    # Inputs:
    - M_0: [batch_size, 3, 3] mass matrix of vertex i
    - M_1: [batch_size, 3, 3] mass matrix of vertex i+1
    - X_0: [batch_size, 3, 1] position of vertex i
    - X_1: [batch_size, 3, 1] position of vertex i+1
    - X_0_init: [batch_size, 3, 1] undeformed position of vertex i
    - X_1_init: [batch_size, 3, 1] undeformed position of vertex i+1

    # Outputs:
    - grad_00: [batch_size, 3, 3] gradient of DX_0 with respect to X_0_init
    - grad_01: [batch_size, 3, 3] gradient of DX_0 with respect to X_1_init
    - grad_10: [batch_size, 3, 3] gradient of DX_1 with respect to X_0_init
    - grad_11: [batch_size, 3, 3] gradient of DX_1 with respect to X_1_init

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """
    batch_size = M_0.shape[0]
    M_0, M_1 = M_0.detach().cpu().numpy(), M_1.detach().cpu().numpy()
    X_0, X_1 = X_0.detach().cpu().numpy(), X_1.detach().cpu().numpy()


    # Compute M_param for each batch
    M_param = np.zeros((batch_size, 3, 3))

    for i in range(batch_size):
        # Skip if any row in M_0[i] or M_1[i] is all zeros
        if np.any(np.all(M_0[i] == 0, axis=1)) or np.any(np.all(M_1[i] == 0, axis=1)):
            continue
        sum_M = M_0[i] + M_1[i]
        if np.linalg.det(sum_M) == 0:
            continue
        M_param[i] = np.linalg.inv(sum_M)

    # Compute Edge and Edge_init for each batch
    Edge = X_1 - X_0  # [batch_size, 3, 1]
    Edge_init = X_1_init - X_0_init  # [batch_size, 3, 1]

    # Compute Edge lengths for each batch
    Edge_length = np.linalg.norm(Edge, axis=1, keepdims=True)  # [batch_size, 1, 1]
    Edge_length_init = np.linalg.norm(Edge_init, axis=1, keepdims=True)  # [batch_size, 1, 1]

    # Compute lambda_param for each batch
    # lambda_param = (Edge_length**2 - Edge_length_init**2) / (Edge_length**2 + Edge_length_init**2)  # [batch_size, 1, 1]
    scale = (4 * Edge_length ** 2 / (Edge_length ** 2 + Edge_length_init ** 2) ** 2)
    scaled_value = scale[:, np.newaxis]

    # Compute gradients for each batch
    grad_00 = np.einsum('bij,bjk->bik', M_1 @ M_param, np.einsum('bi,bj->bij', Edge, Edge_init)) * scaled_value
    grad_01 = -np.einsum('bij,bjk->bik', M_1 @ M_param, np.einsum('bi,bj->bij', Edge, Edge_init)) * scaled_value
    grad_10 = -np.einsum('bij,bjk->bik', M_0 @ M_param, np.einsum('bi,bj->bij', Edge, Edge_init)) * scaled_value
    grad_11 = np.einsum('bij,bjk->bik', M_0 @ M_param, np.einsum('bi,bj->bij', Edge, Edge_init)) * scaled_value

    grad_DX_X_init = np.concatenate(
        (np.concatenate((grad_00, grad_01), axis=2),
         np.concatenate((grad_10, grad_11), axis=2)),
        axis=1
    )

    return grad_DX_X_init


def grad_DX_M_ICitr_batch(M_0, M_1, X_0, X_1, X_0_init, X_1_init, mask):
    """
    Robust batch gradient for inextensibility wrt mass matrix.
    Inputs:
      M_0, M_1:        (B,3,3)
      X_0, X_1:        (B,3,1)
      X_0_init, X_1_init: (B,3,1)
      mask:            (B,) or (B,1) or (B,1,1); True = active edge
    Output:
      grad_DX_X:       (B,6,6)
    """
    # to numpy (no-ops if already np)
    def to_np(x): return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    M_0, M_1 = to_np(M_0), to_np(M_1)                 # (B,3,3)
    X_0, X_1 = to_np(X_0), to_np(X_1)                 # (B,3,1)
    X_0_init, X_1_init = to_np(X_0_init), to_np(X_1_init)

    mask = to_np(mask).astype(bool).reshape(-1)       # (B,)

    B = M_0.shape[0]
    out = np.zeros((B, 6, 2), dtype=np.float64)       # default zeros

    # ---- build “active” mask per batch entry ----
    m0_zero = (np.abs(M_0).sum(axis=(1,2)) == 0)
    m1_zero = (np.abs(M_1).sum(axis=(1,2)) == 0)
    x1_zero = (np.abs(X_1).sum(axis=(1,2)) == 0)
    x1i_zero= (np.abs(X_1_init).sum(axis=(1,2)) == 0)

    # your skip rules:
    # 1) if both M_0 and M_1 are all-zero -> skip
    bothM_zero = m0_zero & m1_zero
    # 2) if M_1, X_1, X_1_init all zero -> skip
    tail_zero = m1_zero & x1_zero & x1i_zero

    active = (~bothM_zero) & (~tail_zero) & mask      # (B,)

    if not np.any(active):
        return out  # all zeros, nothing to do

    idx = np.where(active)[0]

    # ---- compute only for active rows ----
    M0v, M1v = M_0[idx], M_1[idx]
    X0v, X1v = X_0[idx], X_1[idx]
    X0iv, X1iv = X_0_init[idx], X_1_init[idx]

    # safe edge scalars
    eps = 1e-12
    Edge = X1v - X0v                       # (b,3,1)
    Edge_i = X1iv - X0iv
    L  = np.linalg.norm(Edge,  axis=1, keepdims=True)   # (b,1,1)
    L0 = np.linalg.norm(Edge_i,axis=1, keepdims=True)
    denom = L**2 + L0**2
    denom = np.where(denom < eps, eps, denom)
    lam = (L**2 - L0**2) / denom                           # (b,1,1)

    # safe inverse of (M0+M1)
    sumM = M0v + M1v                                        # (b,3,3)
    Mparam = np.linalg.pinv(sumM, rcond=1e-12)              # (b,3,3)
    MM = Mparam @ Mparam

    I = np.broadcast_to(np.eye(3), Mparam.shape)            # (b,3,3)


    # grads (each (b,3,1)) — your earlier formulas
    g00 = -np.einsum('bij,bjk,bkl->bil', M1v, MM, Edge) * lam
    g01 =  np.einsum('bij,bjk->bik', (I - M1v @ Mparam), Mparam @ Edge) * lam
    g10 = -np.einsum('bij,bjk->bik', (I - M0v @ Mparam), Mparam @ Edge) * lam
    g11 =  np.einsum('bij,bjk,bkl->bil', M0v, MM, Edge) * lam

    top = np.concatenate([g00, g01], axis=2)   # (b,3,2)
    bot = np.concatenate([g10, g11], axis=2)   # (b,3,2)
    packed = np.concatenate([top, bot], axis=1)  # (b,6,2)

    out[idx] = packed
    return out


    # Damping
def grad_DX_damping_batch(n_vert, integration_ratio, dt, b_DLOs_velocity, n_branch):
    """
    Batch version of Gradient of the damping constraint iterative function with respect to the damping coefficients.

    # Inputs:
    - integration_ratio: [1] integration ratio for the damping
    - dt: [1] time step size
    - b_DLOs_velocity: [batch_size, n_vert, 3] velocities of the DLOs

    # Outputs:
    - grad_DX_damping: [batch_size, n_vert*3, n_branch] gradient of DX_damping with respect to the damping coefficients

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """

    batch_size = b_DLOs_velocity.shape[0]


    b_DLOs_velocity_np = b_DLOs_velocity.detach().cpu().numpy()
    integration_ratio_np = integration_ratio.detach().cpu().numpy()
    dt_np = dt
    
    # separate the velocities for each branch
    b_DLOs_velocity_expand = np.zeros((batch_size, n_vert, 3, n_branch), dtype=np.float32)
    for i in range(batch_size):
        b_DLOs_velocity_expand[i, :, :, i % n_branch] = b_DLOs_velocity_np[i, :, :]
    b_DLOs_velocity_expand = b_DLOs_velocity_expand.reshape(batch_size, n_vert * 3, n_branch)
    grad_DX_damping = - integration_ratio_np * dt_np**2 * b_DLOs_velocity_expand

    return grad_DX_damping

    # Integration Ratio
def grad_DX_IR_batch(dt, b_DLOs_velocity, mass_matrix, force, damping):
    """
    Batch version of Gradient of the integration ratio constraint iterative function with respect to the integration ratio.

    # Inputs:
    - dt: [1] time step size
    - b_DLOs_velocity: [batch_size, n_vert, 3] velocities of the DLOs
    - mass_matrix: [batch_size, n_vert, 3, 3] mass matrix of the DLOs
    - force: [batch_size, n_vert, 3] forces acting on the DLOs
    - damping: [n_branch] damping forces acting on the DLOs

    # Outputs:
    - grad_DX_IR: [batch_size, n_vert*3, 1] gradient of DX_IR with respect to the integration ratio

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """
    
    batch_size = b_DLOs_velocity.shape[0]
    n_branch = damping.shape

    dt_np = dt
    b_DLOs_velocity_np = b_DLOs_velocity.detach().cpu().numpy()#(1,7,3)
    mass_matrix_np = mass_matrix.detach().cpu().numpy()
    force_np = force.detach().cpu().numpy()#(1,7,3)
    damping_np = damping.detach().cpu().numpy()#(5.0)
    if damping_np.ndim == 0:
        damping_expand = damping_np.reshape(1, 1, 1)*np.ones((batch_size, 1, 1)) #batch_size, 1, 1
    else:
        damping_expand = damping_np.reshape(-1, 1, 1)*np.ones((batch_size, 1, 1))  #batch_size*branch, 1, 1

    M_inv = np.linalg.pinv(mass_matrix_np)  # [batch_size, n_vert, 3, 3]

    acc = np.einsum('ijkl,ijl->ijk', M_inv, force_np) - b_DLOs_velocity_np * damping_expand
    vel = b_DLOs_velocity_np + acc * dt_np
    vel_shrinked = vel.reshape(batch_size, -1, 1)  # [batch_size, n_vert*3, 1]
    grad_DX_IR = vel_shrinked * dt_np

    return grad_DX_IR

def grad_DX_X_ICEC_batch(M_0, M_1):
    """
    Batch version of Gradient of the inextensibility constraint iterative function with respect to the positions X_0 and X_1.

    # Inputs:
    - M_0: [batch*n_parent_branch, n_vertices, 3, 3] mass matrix at two coupling index of parent branch
    - M_1: [batch*n_child_branch, 3, 3] mass matrix of the first index of two children branches


    # Outputs:
    - grad_X_pc_pc: [batch, 3, 3] gradient of DX_pc with respect to parent branch mass matrix at two coupling index M_pc
    - grad_X_pc_cc: [batch, 3, 3] gradient of DX_pc with respect to two children branches mass matrix at two coupling index M_cc
    - grad_X_cc_pc: [batch, 3, 3] gradient of two children branch's DX_cc with respect to parent branch mass matrix M_pc
    - grad_X_cc_cc: [batch, 3, 3] gradient of two children branch's DX_cc with respect to their own mass matrix M_cc

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """

    M_0, M_1 = M_0.detach().cpu().numpy(), M_1.detach().cpu().numpy()
    batch_size = M_0.shape[0]

    # Compute M_param for each batch
    M_param = np.linalg.inv(M_0 + M_1)  # [batch_size, 3, 3]


    grad_00 = -np.einsum('bij,bjk->bik', M_1, M_param)

    grad_01 = np.einsum('bij,bjk->bik', M_1, M_param)

    grad_10 = np.einsum('bij,bjk->bik', M_0, M_param)

    grad_11 = -np.einsum('bij,bjk->bik', M_0, M_param)

    grad_DX_X = np.concatenate(
        (np.concatenate((grad_00, grad_01), axis=2),
         np.concatenate((grad_10, grad_11), axis=2)),
        axis=1
    )

    return grad_00, grad_01, grad_10, grad_11, grad_DX_X

def grad_DX_M_ICEC_batch(M_pc, M_cc, X_pc, X_cc):
    """
    Batch version of Gradient of the inextensibility constraint iterative function with respect to the mass matrices M_0 and M_1.

    # Inputs:
    - M_pc: [batch, n_vertices, 3, 3] mass matrix at two coupling index of parent branch
    - M_cc: [batch, 3, 3] mass matrix of the first index of two children branches
    - X_pc: [batch, 3, 1] position of vertex at two coupling index of parent branch
    - X_cc: [batch, 3, 1] position of vertex of the first index of two children branches

    # Outputs:
    - grad_M_pc_pc: [batch, 3, 1] gradient of DM_pc with respect to parent branch mass matrix at two coupling index M_pc
    - grad_M_pc_cc: [batch, 3, 1] gradient of DM_pc with respect to two children branches mass matrix at two coupling index M_cc
    - grad_M_cc_pc: [batch, 3, 1] gradient of two children branch's DM_cc with respect to parent branch mass matrix M_pc
    - grad_M_cc_cc: [batch, 3, 1] gradient of two children branch's DM_cc with respect to their own mass matrix M_cc

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """

    M_pc, M_cc = M_pc.detach().cpu().numpy(), M_cc.detach().cpu().numpy()
    X_pc, X_cc = X_pc.detach().cpu().numpy(), X_cc.detach().cpu().numpy()

    # Compute M_param for each batch
    M_param = np.linalg.inv(M_pc + M_cc)  # [2, 3, 3]

    # Compute Edge and Edge_init for each batch ## zero-mask not applied yet
    Edge = X_cc - X_pc  # [2, 3, 1]


    # Compute gradients for each batch
    grad_M_pc_pc = -np.einsum('bij,bjk,bkl->bil', M_cc, M_param @ M_param, Edge)
    grad_M_pc_cc = np.einsum('bij,bjk->bik', (np.eye(3) - M_cc @ M_param), (M_param @ Edge))
    grad_M_cc_pc = -np.einsum('bij,bjk->bik', (np.eye(3) - M_pc @ M_param), M_param @ Edge)  # [batch_size, 3, 1]
    grad_M_cc_cc = np.einsum('bij,bjk,bkl->bil', M_pc, M_param @ M_param, Edge)  # [batch_size, 3, 1]

    grad_DX_M = np.concatenate(
        (np.concatenate((grad_M_pc_pc, grad_M_pc_cc), axis=2),
         np.concatenate((grad_M_cc_pc, grad_M_cc_cc), axis=2)),
        axis=1
    )

    
    return grad_M_pc_pc, grad_M_pc_cc, grad_M_cc_pc, grad_M_cc_cc, grad_DX_M