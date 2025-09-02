import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import util
from util import DEFT_initialization, construct_b_DLOs, clamp_index, index_init, save_pickle, Train_DEFTData, \
    Eval_DEFTData
from DEFT_sim import DEFT_sim
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
import glob
import pandas as pd


def train(train_batch, BDLO_type, total_time, train_time_horizon, undeform_vis, inference_vis, inference_1_batch,
          residual_learning, clamp_type, load_model, run_idx):
    # The total_time parameter is the maximum timesteps of the loaded data
    # The train_time_horizon is how many timesteps to unroll the simulation during training
    # The function trains or partially fine-tunes a DEFT model for a specific branched BDLO type

    eval_time_horizon = total_time - 2  # Number of timesteps for evaluation

    # Explanation of notation in the code:
    # - undeformed_BDLO: A tensor containing the initial (undeformed) vertex positions of the branched BDLO
    # - n_parent_vertices / n_child1_vertices / n_child2_vertices: The number of vertices for the main branch, child1, and child2, respectively

    # Prepare BDLO-specific data depending on BDLO_type
    if BDLO_type == 1:
        # Set the undeformed shape of BDLO1 as a tensor of shape [1, 20, 3], then permute to [n_parent_vertices+..., 3]
        undeformed_BDLO = torch.tensor([[[-0.6790, -0.6355, -0.5595, -0.4539, -0.3688, -0.2776, -0.1857,
                                          -0.0991, 0.0102, 0.0808, 0.1357, 0.2081, 0.2404, -0.4279,
                                          -0.4880, -0.5394, -0.5559, 0.0698, 0.0991, 0.1125]],
                                        [[0.0035, -0.0066, -0.0285, -0.0349, -0.0704, -0.0663, -0.0744,
                                          -0.0957, -0.0702, -0.0592, -0.0452, -0.0236, -0.0134, -0.0813,
                                          -0.1233, -0.1875, -0.2178, -0.1044, -0.1858, -0.2165]],
                                        [[0.0108, 0.0104, 0.0083, 0.0104, 0.0083, 0.0145, 0.0133,
                                          0.0198, 0.0155, 0.0231, 0.0199, 0.0154, 0.0169, 0.0160,
                                          0.0153, 0.0090, 0.0121, 0.0205, 0.0155, 0.0148]]]).permute(1, 2, 0)

        # Number of vertices along the parent branch and the two child branches
        n_parent_vertices = 13
        n_child1_vertices = 5
        n_child2_vertices = 4

        # Depending on clamp_type, we set the train/eval dataset sizes and the selection of clamped vertices
        if clamp_type == "ends":
            train_set_number = 77
            eval_set_number = 24
            parent_clamped_selection = torch.tensor((0, 1, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))
        else:
            train_set_number = 71
            eval_set_number = 18
            parent_clamped_selection = torch.tensor((2, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))

        # cs_n_vert holds the number of child1 and child2 vertices
        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        # n_vert is the parent branch vertex count
        n_vert = n_parent_vertices
        # Number of edges in the parent branch is n_vert - 1
        n_edge = n_vert - 1

        # Sanity check: parent branch should have more vertices than any child branch
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Define the stiffness parameters as nn.Parameters for optimization or subsequent usage
        bend_stiffness_parent = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child1 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child2 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device))

        # Damping parameters for each branch
        damping = nn.Parameter(torch.tensor((2.5, 2.5, 2.5), device=device))

        # If we use residual learning, learning_weight is used to scale the residual from the GNN
        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.02, device=device))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device))

        # Indices that define which vertices couple the child branches to the parent
        rigid_body_coupling_index = [4, 8]

        # Mass and moment-of-inertia scaling factors
        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

    if BDLO_type == 2:
        # Similar initialization for BDLO2
        undeformed_BDLO = torch.tensor([
            [[0.0150, 0.0157, 0.0125, 0.0109, 0.0164, 0.0131, 0.0104,
              0.0081, 0.0083, 0.0079, 0.0093, 0.0108, 0.0150, 0.0109,
              0.0116, 0.0110, 0.0111, 0.0084, 0.0103, 0.0097]],
            [[0.1521, 0.1426, 0.1021, 0.0928, 0.0882, 0.0711, 0.0678,
              0.0894, 0.1109, 0.1374, 0.1708, 0.1855, 0.0339, -0.0410,
              -0.1058, -0.1266, 0.0465, -0.0155, -0.0733, -0.0954]],
            [[-0.1706, -0.1409, -0.0875, -0.0018, 0.0702, 0.1685, 0.2583,
              0.3363, 0.3894, 0.4529, 0.5106, 0.5355, 0.0704, 0.0615,
              0.0093, -0.0080, 0.3405, 0.3217, 0.2929, 0.2834]]
        ]).permute(1, 2, 0)

        n_parent_vertices = 12
        n_child1_vertices = 5
        n_child2_vertices = 5
        train_set_number = 110
        eval_set_number = 37

        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        n_vert = n_parent_vertices
        n_edge = n_vert - 1
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Stiffness parameters
        bend_stiffness_parent = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child1 = nn.Parameter(1.5e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child2 = nn.Parameter(1.5e-3 * torch.ones((1, 1, n_edge), device=device))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device))

        # Damping parameters
        damping = nn.Parameter(torch.tensor((2.5, 2., 2.), device=device))

        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.02, device=device))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device))

        # Rigid body coupling index: the parent-children connection points
        rigid_body_coupling_index = [4, 7]

        # Mass, MOI scaling, etc.
        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

        # Which vertices are clamped in the dataset
        parent_clamped_selection = torch.tensor((0, 1, -2, -1))
        child1_clamped_selection = torch.tensor((2))
        child2_clamped_selection = torch.tensor((2))

    if BDLO_type == 3:
        # Initialization for BDLO3
        undeformed_BDLO = torch.tensor([
            [[0.0099, 0.0114, 0.0109, 0.0084, 0.0130, 0.0143, 0.0119,
              0.0133, 0.0135, 0.0136, 0.0136, 0.0151, 0.0124, 0.0093,
              0.0120, 0.0132, 0.0121]],
            [[-0.0444, -0.0684, -0.1235, -0.1722, -0.1973, -0.2265, -0.2232,
              -0.1956, -0.1675, -0.1150, -0.0544, -0.0249, -0.2632, -0.3340,
              -0.2580, -0.3370, -0.3594]],
            [[-0.5656, -0.5434, -0.4977, -0.4399, -0.3552, -0.2506, -0.1563,
              -0.0530, 0.0092, 0.0709, 0.1222, 0.1390, -0.3781, -0.4082,
              -0.0169, -0.0347, -0.0420]]
        ]).permute(1, 2, 0)

        n_parent_vertices = 12
        n_child1_vertices = 3
        n_child2_vertices = 4

        if clamp_type == "ends":
            train_set_number = 103
            eval_set_number = 26
            parent_clamped_selection = torch.tensor((0, 1, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))
        else:
            train_set_number = 39
            eval_set_number = 12
            parent_clamped_selection = torch.tensor((3, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))

        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        n_vert = n_parent_vertices
        n_edge = n_vert - 1
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Stiffness parameters
        bend_stiffness_parent = nn.Parameter(2.5e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child1 = nn.Parameter(2.5e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child2 = nn.Parameter(2.5e-3 * torch.ones((1, 1, n_edge), device=device))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device))
        damping = nn.Parameter(torch.tensor((2., 2., 2.), device=device))

        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.02, device=device))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device))

        rigid_body_coupling_index = [4, 7]
        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

    if BDLO_type == 4:
        # Initialization for BDLO4
        undeformed_BDLO = torch.tensor([
            [[0.0108, 0.0122, 0.0112, 0.0116, 0.0116, 0.0169, 0.0122,
              0.0198, 0.0173, 0.0140, 0.0152, 0.0156, 0.0120, 0.0107,
              0.0163, 0.0154]],
            [[-0.1680, -0.1938, -0.2439, -0.2991, -0.3230, -0.3345, -0.3376,
              -0.3248, -0.3100, -0.2727, -0.2182, -0.1878, -0.3922, -0.4643,
              -0.3866, -0.4500]],
            [[-0.5774, -0.5491, -0.4909, -0.4085, -0.3219, -0.2371, -0.1568,
              -0.0645, 0.0231, 0.0828, 0.1411, 0.1664, -0.3430, -0.3652,
              0.0434, 0.0658]]
        ]).permute(1, 2, 0)

        n_parent_vertices = 12
        n_child1_vertices = 3
        n_child2_vertices = 3
        train_set_number = 74
        eval_set_number = 25

        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        n_vert = n_parent_vertices
        n_edge = n_vert - 1
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Stiffness parameters
        bend_stiffness_parent = nn.Parameter(3e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child1 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child2 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device))

        # Damping for each of the 3 branches
        damping = nn.Parameter(torch.tensor((3., 4, 4.), device=device))

        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.02, device=device))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device))

        # Connection indices for the child branches
        rigid_body_coupling_index = [4, 8]

        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

        parent_clamped_selection = torch.tensor((0, 1, -2, -1))
        child1_clamped_selection = torch.tensor((2))
        child2_clamped_selection = torch.tensor((2))

    # Decide how many batches we use in evaluation (1 batch if inference_1_batch is True, else the entire eval_set_number)
    if inference_1_batch:
        eval_batch = 1
    else:
        eval_batch = eval_set_number

    # Number of vertices in the child branches
    n_children_vertices = (n_child1_vertices, n_child2_vertices)

    # Extract parent and child vertices from the undeformed BDLO
    parent_vertices_undeform = undeformed_BDLO[:, :n_parent_vertices]
    child1_vertices_undeform = undeformed_BDLO[:, n_parent_vertices: n_parent_vertices + n_children_vertices[0] - 1]
    child2_vertices_undeform = undeformed_BDLO[:, n_parent_vertices + n_children_vertices[0] - 1:]

    # DEFT_initialization returns scaled mass, MOI, rod orientations, nominal length, etc.
    b_DLO_mass, parent_MOI, children_MOI, parent_rod_orientation, children_rod_orientation, b_nominal_length = DEFT_initialization(
        parent_vertices_undeform,
        child1_vertices_undeform,
        child2_vertices_undeform,
        n_branch,
        n_parent_vertices,
        cs_n_vert,
        rigid_body_coupling_index,
        parent_mass_scale,
        parent_moment_scale,
        children_moment_scale,
        children_mass_scale,
        moment_ratio
    )

    # Construct the branched BDLO from data for the training batch
    # b_DLOs_vertices_undeform_untransform is the original set of undeformed states across the batch
    # The second return is an optional placeholder for transformations or expansions
    b_DLOs_vertices_undeform_untransform, _ = construct_b_DLOs(
        train_batch,
        rigid_body_coupling_index,
        n_parent_vertices,
        cs_n_vert,
        n_branch,
        parent_vertices_undeform,
        parent_vertices_undeform,
        child1_vertices_undeform,
        child1_vertices_undeform,
        child2_vertices_undeform,
        child2_vertices_undeform
    )

    # Transform the axis from local coordinate to global by re-indexing the coordinate axes
    b_DLOs_vertices_undeform_transform = torch.zeros_like(b_DLOs_vertices_undeform_untransform)
    b_DLOs_vertices_undeform_transform[:, :, :, 0] = -b_DLOs_vertices_undeform_untransform[:, :, :, 2]
    b_DLOs_vertices_undeform_transform[:, :, :, 1] = -b_DLOs_vertices_undeform_untransform[:, :, :, 0]
    b_DLOs_vertices_undeform_transform[:, :, :, 2] = b_DLOs_vertices_undeform_untransform[:, :, :, 1]

    # The first sample in the batch of undeformed vertices (reshape to [n_branch, n_vert, 3])
    b_undeformed_vert = b_DLOs_vertices_undeform_transform[0].view(n_branch, -1, 3)

    # rdm_scale = 0.01
    #
    # b_undeformed_vert, b_DLO_mass = randomize_input(n_branch,n_parent_vertices,rdm_scale,b_undeformed_vert, b_DLO_mass)

    # Initialize index selection for parent MOI indices, etc.
    index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(
        rigid_body_coupling_index,
        n_branch
    )

    # Decide which vertices get clamped (i.e., fixed in space/rotation) for the training and evaluation sets
    clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(
        train_batch,
        parent_clamped_selection,
        child1_clamped_selection,
        child2_clamped_selection,
        n_branch,
        n_parent_vertices,
        clamp_parent,
        clamp_child1,
        clamp_child2
    )
    eval_clamped_index, eval_parent_theta_clamp, eval_child1_theta_clamp, eval_child2_theta_clamp = clamp_index(
        eval_batch,
        parent_clamped_selection,
        child1_clamped_selection,
        child2_clamped_selection,
        n_branch,
        n_parent_vertices,
        clamp_parent,
        clamp_child1,
        clamp_child2
    )

    # Timestep for simulation
    dt = 0.01

    # Instantiate DEFT_sim objects for training and evaluation
    DEFT_sim_train = DEFT_sim(
        batch=train_batch,
        n_branch=n_branch,
        n_vert=n_vert,
        cs_n_vert=cs_n_vert,
        b_init_n_vert=b_undeformed_vert,
        n_edge=n_vert - 1,
        b_undeformed_vert=b_undeformed_vert,
        b_DLO_mass=b_DLO_mass,
        parent_DLO_MOI=parent_MOI,
        children_DLO_MOI=children_MOI,
        device=device,
        clamped_index=clamped_index,
        rigid_body_coupling_index=rigid_body_coupling_index,
        parent_MOI_index1=parent_MOI_index1,
        parent_MOI_index2=parent_MOI_index2,
        parent_clamped_selection=parent_clamped_selection,
        child1_clamped_selection=child1_clamped_selection,
        child2_clamped_selection=child2_clamped_selection,
        clamp_parent=clamp_parent,
        clamp_child1=clamp_child1,
        clamp_child2=clamp_child2,
        index_selection1=index_selection1,
        index_selection2=index_selection2,
        bend_stiffness_parent=bend_stiffness_parent,
        bend_stiffness_child1=bend_stiffness_child1,
        bend_stiffness_child2=bend_stiffness_child2,
        twist_stiffness=twist_stiffness,
        damping=damping,
        learning_weight=learning_weight
    )

    # Load pretrained models for initialization depending on BDLO_type and clamp_type
    if load_model:
        if BDLO_type == 1 and clamp_type == "ends":
            DEFT_sim_train.load_state_dict(torch.load("save_model/BDLO1/DEFT_1_780_1.pth"), strict=False)
        if BDLO_type == 1 and clamp_type == "middle":
            DEFT_sim_train.load_state_dict(torch.load("save_model/BDLO1/DEFT_middle_1_2260_1.pth"), strict=False)
        if BDLO_type == 2:
            DEFT_sim_train.load_state_dict(torch.load("save_model/BDLO2/DEFT_2_820_2.pth"), strict=False)
        if BDLO_type == 3:
            DEFT_sim_train.load_state_dict(torch.load("save_model/BDLO3/DEFT_3_40_3.pth"), strict=False)
        if BDLO_type == 3 and clamp_type == "middle":
            DEFT_sim_train.load_state_dict(torch.load("save_model/BDLO3/DEFT_middle_3_2320_1.pth"), strict=False)
        if BDLO_type == 4:
            DEFT_sim_train.load_state_dict(torch.load("save_model/BDLO4/DEFT_4_40_3.pth"), strict=False)

    # If we want to visualize the undeformed states
    if undeform_vis:
        # Visualize the first batch's undeformed state
        first_batch_vertices = b_DLOs_vertices_undeform_transform[0]  # shape: [3, n_vert, 3]
        colors = ['red', 'green', 'blue']

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # For each branch, plot all vertex positions (here, just one set since it's "undeformed_vis")
        for branch_idx in range(3):
            branch_positions = first_batch_vertices[branch_idx, :, :]

            for vertex_idx in range(branch_positions.shape[0]):
                vertex_positions = branch_positions[vertex_idx, :]
                x = vertex_positions[0].unsqueeze(dim=0).numpy()
                y = vertex_positions[1].unsqueeze(dim=0).numpy()
                z = vertex_positions[2].unsqueeze(dim=0).numpy()
                ax.scatter(x, y, z, color=colors[branch_idx], alpha=1.)

        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(-0.5, 1.0)
        ax.set_zlim(-0.25, 1.25)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('Trajectories of Vertices Over Time (First Batch)')

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f'Branch {i}') for i in range(3)]
        ax.legend(handles=legend_elements)
        plt.show()

    # Scale factor for learning rate
    lr_scale = 10

    # Define loss function
    loss_func = torch.nn.MSELoss()

    # We separate parameter sets based on whether we are doing residual learning or not
    gnn_params = DEFT_sim_train.GNN_tree.parameters()

    if not residual_learning:
        # If not using residual learning, we optimize all or most DEFT parameters
        parameters_to_update = [
            {"params": DEFT_sim_train.p_DLO_diagonal, "lr": 1e-5 * lr_scale},
            {"params": DEFT_sim_train.c_DLO_diagonal, "lr": 1e-5 * lr_scale},
            {"params": DEFT_sim_train.integration_ratio, "lr": 1e-5 * lr_scale},
            {"params": DEFT_sim_train.velocity_ratio, "lr": 1e-5 * lr_scale},
            {"params": DEFT_sim_train.undeformed_vert, "lr": 1e-5 * lr_scale},
            {"params": DEFT_sim_train.mass_diagonal, "lr": 1e-5 * lr_scale},
            {"params": DEFT_sim_train.damping, "lr": 1e-5 * lr_scale},
            {"params": DEFT_sim_train.gravity, "lr": 1e-5 * lr_scale},
            {"params": DEFT_sim_train.DEFT_func.twist_stiffness, "lr": 1e-5 * lr_scale},
            {"params": DEFT_sim_train.DEFT_func.bend_stiffness_parent, "lr": 1e-9 * lr_scale},
            {"params": DEFT_sim_train.DEFT_func.bend_stiffness_child1, "lr": 1e-9 * lr_scale},
            {"params": DEFT_sim_train.DEFT_func.bend_stiffness_child2, "lr": 1e-9 * lr_scale},
        ]
    else:
        # If using residual learning, we mainly update the GNN and the residual learning weight
        parameters_to_update = [
            {"params": DEFT_sim_train.learning_weight, "lr": 1e-6 * lr_scale},
            {"params": gnn_params, "lr": 1e-5 * lr_scale}
        ]

    # Define the optimizer (here, SGD) with the chosen parameters
    optimizer = optim.SGD(parameters_to_update)

    # We'll store evaluation results after certain intervals
    eval_epochs = []
    eval_losses = []

    # We'll store training results as well
    training_epochs = []
    training_losses = []

    # Define number of epochs to train
    train_epoch = 100
    save_steps = 0
    evaluate_period = 20
    model = "DEFT"
    training_case = 1

    # The main training loop
    if model == "DEFT":
        # Get the input data from the loader
        vis = False
        vis_type = "DEFT_%s" % BDLO_type
        previous_b_DLOs_vertices_traj, b_DLOs_vertices_traj, target_b_DLOs_vertices_traj = get_data(BDLO_type, train_time_horizon, total_time, n_parent_vertices, n_children_vertices, rigid_body_coupling_index, n_branch, DEFT_sim_train.batch)

        # Forward pass through the DEFT model for train_time_horizon timesteps
        traj_loss, total_loss = DEFT_sim_train.iterative_sim(
            train_time_horizon,
            b_DLOs_vertices_traj,
            previous_b_DLOs_vertices_traj,
            target_b_DLOs_vertices_traj,
            loss_func,
            dt,
            parent_theta_clamp,
            child1_theta_clamp,
            child2_theta_clamp,
            inference_1_batch,
            run_idx,
            vis_type=vis_type,
            vis=vis
        )

def get_data(BDLO_type, train_time_horizon, total_time, n_parent_vertices, n_children_vertices, rigid_body_coupling_index, n_branch, batch_size):
    root_dir = "dataset/BDLO%s/train/" % BDLO_type
    file_list = glob.glob(root_dir + "*")
    verts = torch.tensor(pd.read_pickle(r'%s' % file_list[0])).view(3, total_time, -1).permute(1, 2, 0)
    
    n_child1_vertices, n_child2_vertices = n_children_vertices
    parent_vertices = verts[:, :n_parent_vertices]
    child1_vertices = verts[:, n_parent_vertices: n_parent_vertices + n_child1_vertices - 1]
    child2_vertices = verts[:, n_parent_vertices + n_child1_vertices - 1:]

    BDLO_vert_no_trans = util.construct_BDLOs_data(total_time, rigid_body_coupling_index,
                                              n_parent_vertices, n_children_vertices,
                                              n_branch, parent_vertices, child1_vertices, child2_vertices)

    BDLO_vert = torch.zeros_like(BDLO_vert_no_trans)
    BDLO_vert[:, :, :, 0] = -BDLO_vert_no_trans[:, :, :, 2]
    BDLO_vert[:, :, :, 1] = -BDLO_vert_no_trans[:, :, :, 0]
    BDLO_vert[:, :, :, 2] = BDLO_vert_no_trans[:, :, :, 1]

    BDLOs_previous_vertices_np = BDLO_vert[0: train_time_horizon].numpy()
    BDLOs_vertices_np = BDLO_vert[1: train_time_horizon + 1].numpy()
    BDLOs_target_vertices_np = BDLO_vert[2: train_time_horizon + 2].numpy()

    BDLOs_previous_vertices = torch.tensor(BDLOs_previous_vertices_np, dtype=torch.float64).repeat(batch_size, 1, 1, 1, 1)
    BDLOs_vertices = torch.tensor(BDLOs_vertices_np, dtype=torch.float64).repeat(batch_size, 1, 1, 1, 1)
    BDLOs_target_vertices = torch.tensor(BDLOs_target_vertices_np, dtype=torch.float64).repeat(batch_size, 1, 1, 1, 1)

    return BDLOs_previous_vertices, BDLOs_vertices, BDLOs_target_vertices

def randomize_input(n_branch, n_vert, rdm_scale, b_undeformed_vert, b_DLO_mass):
    '''
    for randomizing input
    :param batch:
    :param n_branch:
    :param n_vert:
    :param rdm_scale:
    :param b_undeformed_vert:
    :param b_DLO_mass:
    :return:
    '''


    # reset random seed each time -> ensures new randomness
    torch.seed()
    zero_vertices_mask = (b_undeformed_vert != 0).to(torch.uint8)
    rdm_vec = torch.rand(n_branch, n_vert, 3, device=device)
    print('rdm_vec',rdm_vec)
    rdm_vec = rdm_vec / rdm_vec.norm(dim=-1, keepdim=True) * rdm_scale * zero_vertices_mask
    b_undeformed_vert = b_undeformed_vert + rdm_vec
    rand_mass = torch.rand(n_branch, n_vert, device=device)
    row_mask = zero_vertices_mask[..., 0]
    rand_mass =row_mask * rand_mass
    print('random_mass',rand_mass)

    b_DLO_mass = b_DLO_mass+rand_mass


    return b_undeformed_vert, b_DLO_mass

if __name__ == "__main__":
    # Setting up a command-line interface for hyperparameters and options

    # Make sure to use double precision for stability in the DEFT simulations
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)

    # Limit the number of threads used by PyTorch and underlying libraries for reproducibility
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # BDLO_type controls which BDLO dataset (and initial parameter sets) to use
    parser.add_argument("--BDLO_type", type=int, default=1)

    # clamp_type indicates how the BDLO is clamped (ends or middle)
    parser.add_argument("--clamp_type", type=str, default="ends")

    # total_time is the maximum number of timesteps we have in the dataset (e.g. 500)
    parser.add_argument("--total_time", type=int, default=500)

    # train_time_horizon is how many timesteps we simulate in each training iteration
    parser.add_argument("--train_time_horizon", type=int, default=10)

    # Whether to visualize the initial undeformed vertices
    parser.add_argument("--undeform_vis", type=bool, default=False)

    # Whether we do inference only for 1 batch (for speed) or for all eval sets
    parser.add_argument("--inference_1_batch", type=bool, default=False)

    # Whether to enable residual learning: if True, GNN-based updates are used
    parser.add_argument("--residual_learning", type=bool, default=False)

    # Training batch size
    parser.add_argument("--train_batch", type=int, default=5)

    # Whether to visualize inference results (for debugging)
    parser.add_argument("--inference_vis", type=bool, default=False)

    # load trained model
    parser.add_argument("--load_model", type=bool, default=True)

    # Flags for which branches are clamped
    clamp_parent = True
    clamp_child1 = False
    clamp_child2 = False

    # Number of branches for the BDLO (1 parent branch + 2 children branches)
    n_branch = 3

    # For simplicity, everything is done on CPU in this version
    device = "cpu"

    args = parser.parse_args()

    for run_idx in range(1):
        print(f"\n=== Run {run_idx + 1}/10 ===")


        # Call the training function with the user-specified arguments
        train(
            train_batch=args.train_batch,
            BDLO_type=args.BDLO_type,
            total_time=args.total_time,
            train_time_horizon=args.train_time_horizon,
            undeform_vis=args.undeform_vis,
            inference_vis=args.inference_vis,
            inference_1_batch=args.inference_1_batch,
            residual_learning=args.residual_learning,
            clamp_type=args.clamp_type,
            load_model=args.load_model,
            run_idx=run_idx
        )
