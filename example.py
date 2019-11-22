from models import my_gru
import numpy as np
import torch
import torch.nn as nn

# input size, hidden neurons, input/output layer dropouts, bidrect
model = my_gru(10, 5, 0.0, 0.0, False)
model.double() # if you want to use float64 over 32
# toggle cuda
if torch.cuda.is_available():
    model.cuda()


sequence = torch.randn(1,5,10).double().cuda()
print(sequence.shape)
hidden = model.init_hidden() # zeros

# run GRU on 1 train example
outputs = model(sequence, hidden)

print(outputs)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=.0001)

# 5 train examples
def train():
    model.train()
    total_loss = 0
    for i in range(5):
        seq = torch.randn(1,5,10).double().cuda()
        hidden = model.init_hidden()

        outputs, hid = model(sequence, hidden)
        pred = outputs[-1][-1][-1] # take output from last unroll of GRU
        label = torch.randn(1)[-1].double().cuda()

        loss = criterion(pred, label)
        loss.backward() #backprop
                
        optimizer.step()

        total_loss += loss.item()

def test():
    model.eval()
    with torch.no_grad(): # speed up testing!
        test_data = torch.randn(1,5,10).double().cuda()
        hidden = model.init_hidden()

        outputs, hid = model(test_data, hidden)

        pred = outputs[-1][-1][-1]
        label = torch.randn(1)[-1].double().cuda()




train()
test()
