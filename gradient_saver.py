import numpy as np

class BackwardGradientIC:
    def __init__(self, batch, num_vertices):
        self.grad_DX_X = None
        self.grad_DX_Xinit = None
        self.grad_DX_M = None
        self.reset(batch, num_vertices)
        return

    def reset(self, batch, num_vertices):
        self.grad_DX_X = np.zeros((batch,num_vertices*3, num_vertices*3), dtype=np.float32)
        self.grad_DX_Xinit = np.zeros((batch,num_vertices*3, num_vertices*3), dtype=np.float32)
        self.grad_DX_M = np.zeros((batch,num_vertices*3, num_vertices), dtype=np.float32)
        return