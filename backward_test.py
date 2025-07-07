import torch
from torch import nn
from torch.autograd import Function
import time

class custom_mlp(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(custom_mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    x_1 = [0]
    y_1 = [0]
    x_2 = [0, 0, 0]
    y_2 = [0, 0, 0]

    model = custom_mlp(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x1_tensor = torch.tensor(x_1, dtype=torch.float32).view(-1, 1)
    y1_tensor = torch.tensor(y_1, dtype=torch.float32).view(-1, 1)
    x2_tensor = torch.tensor(x_2, dtype=torch.float32).view(-1, 1)
    y2_tensor = torch.tensor(y_2, dtype=torch.float32).view(-1, 1)
    
    for i in range(5):
        # 1 input, 1 output
        output1 = model(x1_tensor)
        loss1 = nn.MSELoss()(output1, y1_tensor)
        
        t0 = time.time()
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        t1 = time.time()
        print(f"Time taken for first backward pass: {t1 - t0:.6f} seconds")

        # 3 inputs, 3 outputs
        output2 = model(x2_tensor)
        loss2 = nn.MSELoss()(output2, y2_tensor)
        
        t2 = time.time()
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()
        t3 = time.time()
        print(f"Time taken for second backward pass: {t3 - t2:.6f} seconds")

        # 1 input, 3 outputs
        mid_output1 = model(x1_tensor)
        mid_output2 = model(mid_output1)
        output3 = model(mid_output2)
        loss31 = nn.MSELoss()(mid_output1, y1_tensor)
        loss32 = nn.MSELoss()(mid_output2, y1_tensor)
        loss33 = nn.MSELoss()(output3, y1_tensor)
        loss3 = loss31 + loss32 + loss33

        t4 = time.time()
        optimizer.zero_grad()
        loss3.backward()
        optimizer.step()
        t5 = time.time()
        print(f"Time taken for third backward pass: {t5 - t4:.6f} seconds")

        # 1 input, 3 outputs, detached
        mid_output1 = model(x1_tensor)
        mid_output2 = model(mid_output1.detach())
        output4 = model(mid_output2.detach())
        loss41 = nn.MSELoss()(mid_output1, y1_tensor)
        loss42 = nn.MSELoss()(mid_output2, y1_tensor)
        loss43 = nn.MSELoss()(output4, y1_tensor)
        loss4 = loss41 + loss42 + loss43

        t6 = time.time()
        optimizer.zero_grad()
        loss4.backward()
        optimizer.step()
        t7 = time.time()
        print(f"Time taken for fourth backward pass (detached): {t7 - t6:.6f} seconds")

        print("----------------------")