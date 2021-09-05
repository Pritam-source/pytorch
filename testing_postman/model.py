import numpy as np
import torch
import torch.nn as nn

X=torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)
n_samples,n_features=X.shape
test=torch.tensor([6],dtype=torch.float32)
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearRegression,self).__init__()
        self.linear=nn.Linear(input_size,output_size)

    def forward(self,x):
        return self.linear(x)

model=LinearRegression(n_features,n_features)

learning_rate=0.01
n_iters=100
loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_iters):
    y_pred=model(X)
    loss=loss_fn(y_pred,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        [w,b]=model.parameters()
        print(f'epoch{epoch+1}:w={w[0][0].item():.3f}, los={loss:.8f}')

print(f'prediction after training: f(6)={model(test).item():.3f}')

