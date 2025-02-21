# DEFT: Differentiable Branched Discrete Elastic Rods for Modeling Furcated DLOs in Real-Time

This repository contains the source code for the paper [DEFT: Differentiable Branched Discrete Elastic Rods for Modeling Furcated DLOs in Real-Time](https://arxiv.org/abs/2406.05931). Project page is [here](https://roahmlab.github.io/DEFT/).

## Introduction
<p align="center">
  <img height="295" width=1200" src="/demo_image.png"/>
</p>

The figures above illustrate how DEFT can be used to autonomously perform a wire insertion task.

**Left:** The system first plans a shape-matching motion, transitioning the BDLO from its initial configuration to the target shape (contoured with yellow), which serves as an intermediate waypoint.

**Right:** Starting from the intermediate configuration, the system performs thread insertion, guiding the BDLO into the target hole while also matching the target shape. Notably, DEFT predicts the shape of the wire recursively without relying on ground truth or perception data at any point in the process.

**Contributions:** While existing research has made progress in modeling single-threaded Deformable Linear Objects (DLOs), extending these approaches to Branched Deformable Linear Objects (BDLOs) presents fundamental challenges. 
The junction points in BDLOs create complex force interactions and strain propagation patterns that cannot be adequately captured by simply connecting multiple single-DLO models.
To address these challenges, this paper presents Differentiable discrete branched Elastic rods for modeling Furcated DLOs in real-Time (DEFT), a novel framework that combines a differentiable physics-based model with a learning framework to: 1) accurately model BDLO dynamics, including dynamic propagation at junction points and grasping in the middle of a BDLO, 2) achieve efficient computation for real-time inference, and 3) enable planning to demonstrate dexterous BDLO manipulation. To the best of our knowledge, this is the first publicly available dataset and code for modeling BDLOs.

**Authors:** Yizhou Chen (yizhouch@umich.edu),  Xiaoyue Wu (wxyluna@umich.edu), Yeheng Zong (yehengz@umich.edu), Anran Li (anranli@umich.edu ), Yuzhen Chen (yuzhench@umich.edu), Julie Wu (jwuxx@umich.edu), Bohao Zhang (jimzhang@umich.edu) and Ram Vasudevan (ramv@umich.edu).

All authors are affiliated with the Robotics department and the department of Mechanical Engineering of the University of Michigan, 2505 Hayward Street, Ann Arbor, Michigan, USA.

## Modeling Results Visualization
<p align="center">
  <img height="530" width=1200" src="/modeling_demo.png"/>
</p>
<p align="center">
  <img height="530" width=1200" src="/modeling_demo2.png"/>
</p>
Visualization of the predicted trajectories for BDLO 1 under two manipulation scenarios, using DEFT, a DEFT ablation that leaves out the constraint described in Theorem 4, and Tree-LSTM. The ground-truth initial position of the vertices are colored in blue, the ground-truth final position of the vertices are colored in pink, and the gradient between these two colors is used to denote the ground truth location over time. 
The predicted vertices are colored as green circles (DEFT), orange circles (DEFT ablation), and light red circles (Tree-LSTM), respectively.
A gradient is used for these predictions to depict the evolution of time, starting from dark and going to light.
Note that the ground truth is only provided at t=0s and prediction is constructed until t=8s.
The prediction is performed recursively, without requiring additional ground-truth data or perception inputs throughout the entire process.

## Dependency 
- Run `pip install -r requirements.txt` to collect all python dependencies.

## Train DEFT Models
Example: To train a DEFT model using the BDLO1 dataset with end-effectors that grasp the BDLO's ends, run the following command: python DEFT_train.py --BDLO_type="1" --clamp_type="ends"

## Dataset
- For each BDLO, dynamic trajectory data is captured in real-world settings using a motion capture system operating at 100 Hz when robots grasp the BDLO’s ends. For details on dataset usage, please refer to DEFT_train.py.
- For BDLO 1 and BDLO 3, we record dynamic trajectory data when one robot grasps the middle of the BDLO while the other robot grasps one of its ends.

## Citation (To be updated)
If you use DEFT in an academic work, please cite using the following BibTex entry:
```
@misc{chen2024differentiable,
      title={Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects}, 
      author={Yizhou Chen and Yiting Zhang and Zachary Brei and Tiancheng Zhang and Yuzhen Chen and Julie Wu and Ram Vasudevan},
      year={2024},
      eprint={2406.05931},
      archivePrefix={arXiv},
      primaryClass={id='cs.RO' full_name='Robotics' is_active=True alt_name=None in_archive='cs' is_general=False description='Roughly includes material in ACM Subject Class I.2.9.'}
}
```


