# Libraries
import numpy as np
import torch
import torch.nn as nn

# ---- Data Loading ----


# ---- Model Architecture ----

# Linear Regression Architecture
class LR(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_features, 1)
        )

    def forward(self, data):
        prediction = self.linear_stack(data)
        return prediction


# Instance, Loss Function, Optimizer

model = LR() # Specify number of features



loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-2)

# ---- Model Training ----