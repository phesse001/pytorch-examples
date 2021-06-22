import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# this is our weight, it requires a gradient since we will be updating it based on the loss
w = torch.tensor([1.0], requires_grad=True) 

def forward(x):
  return x * w

def loss(y_pred, y_val):
  return (y_pred - y_val) ** 2

# before training
print("Prediction (before training)", 4, forward(4).item())

for epoch in range(10):
  for x_val, y_val in zip(x_data, y_data):
    y_pred = forward(x_val)
    l = loss(y_pred, y_val)
    l.backward()
    print("\tgrad: ", x_val, y_val, w.grad.item()) # dloss/dw is stored in w.grad.item()
    # take small step of learning rate to change weight for better prediction
    # w.data stores that value of w
    w.data = w.data - 0.01 * w.grad.item()
    # zero the gradients after updaing weights
    w.grad.data.zero_()

  print(f"Epoch: {epoch} | Loss: {l.item()}")

# after training
print("Prediction (after training)", 4, forward(4).item())
