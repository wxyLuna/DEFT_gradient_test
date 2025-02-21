# DEFT: Differentiable Branched Discrete Elastic Rods for Modeling Furcated DLOs in Real-Time

This repository contains the source code for the paper [DEFT: Differentiable Branched Discrete Elastic Rods for Modeling Furcated DLOs in Real-Time](https://arxiv.org/abs/2406.05931).

<p align="center">
  <img height="330" src="/demo_image.png"/>
</p>
This paper introduces DEFT, a novel framework that combines a differentiable physics-based model with a learning-based approach to accurately model and predict the dynamic behavior of Branched Deformable Linear Objects (BDLOs) in real time. As this paper illustrates, this model can be used in concert with a motion planning algorithm to autonomously manipulate BDLOs. The figures above illustrate how DEFT can be used to autonomously perform a wire insertion task.

**Left:** The system first plans a shape-matching motion, transitioning the BDLO from its initial configuration to the target shape (contoured with yellow), which serves as an intermediate waypoint.

**Right:** Starting from the intermediate configuration, the system performs thread insertion, guiding the BDLO into the target hole while also matching the target shape. Notably, DEFT predicts the shape of the wire recursively without relying on ground truth or perception data at any point in the process.

## Introduction
While existing research has made progress in modeling single-threaded Deformable Linear Objects (DLOs), extending these approaches to Branched Deformable Linear Objects (BDLOs) presents fundamental challenges. 
The junction points in BDLOs create complex force interactions and strain propagation patterns that cannot be adequately captured by simply connecting multiple single-DLO models.
To address these challenges, this paper presents Differentiable discrete branched Elastic rods for modeling Furcated DLOs in real-Time (\DEFT), a novel framework that combines a differentiable physics-based model with a learning framework to: 1) accurately model BDLO dynamics, including dynamic propagation at junction points and grasping in the middle of a BDLO, 2) achieve efficient computation for real-time inference, and 3) enable planning to demonstrate dexterous BDLO manipulation. 

**Authors:** Yizhou Chen (yizhouch@umich.edu),  (yitzhang@umich.edu ), Zachary Brei (breizach@umich.edu), Tiancheng Zhang (zhangtc@umich.edu ), Yuzhen Chen (yuzhench@umich.edu), Julie Wu (jwuxx@umich.edu) and Ram Vasudevan (ramv@umich.edu).

All authors are affiliated with the Robotics department and the department of Mechanical Engineering of the University of Michigan, 2505 Hayward Street, Ann Arbor, Michigan, USA.
