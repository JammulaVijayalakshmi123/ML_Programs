import numpy as np
# Define the sigmoid function
def sigmoid(x):
 return 1 / (1 + np.exp(-x))
# Derivative of the sigmoid function
def ds(x):
 return x * (1 - x)
# Input data
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
# Normalize input and output
x = x / np.amax(x, axis=0)
y = y / 100
# Parameters
epoch = 5
lr = 0.1
iln = 2
oln = 1
hln = 1
# Initialize weights and biases
wh = np.random.uniform(size=(iln, hln))
bh = np.random.uniform(size=(1, hln))
wout = np.random.uniform(size=(hln, oln))
bout = np.random.uniform(size=(1, oln))
# Training loop
for i in range(epoch):
# Forward pass
 hinp = np.dot(x, wh) + bh
 hlayer_act = sigmoid(hinp)
 oinp = np.dot(hlayer_act, wout)
 output = sigmoid(oinp)
 # Backpropagation
 eo = y - output
 o_grad = ds(output)
 d_output = eo * o_grad
 eh = d_output.dot(wout.T)
 h_grad = ds(hlayer_act)
 d_hidden = eh * h_grad
 # Update weights and biases
 wh += x.T.dot(d_hidden) * lr
 wout += hlayer_act.T.dot(d_output) * lr
 # Print outputs for each epoch
 print("Epoch:", i + 1)
 print("Expected Output:")
 print(y)
 print("Predicted Output:")
 print(output)
 print()
