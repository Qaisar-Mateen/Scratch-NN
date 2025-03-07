import numpy as np
import matplotlib.pyplot as plt
from neural_network import *
from f_load_dataset import load_dataset

# load dataset
train_x, train_t, val_x, val_t, test_x, test_t = load_dataset()

# create l-dim network by just adding num of neurons in layer_dim
# first and last elements represent input and output layers dim
# layer_dim = [1, 50, 50, 50, 50, 50, 50, 50, 50, 1]# 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 1]

layer_dim = [1, 50, 50, 50, 50, 50, 1]
            #  50, 50, 50, 50, 50, 1]# 50, 50, 50, 50, 50, 1]

# add activation functions name here. 
# input layer activation function is None
# activations = [None, 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'identity']

activations = [None, 'lrelu', 'lrelu', 'lrelu', 'lrelu', 'lrelu', # 'lrelu', 'lrelu', 'lrelu', 'lrelu', 'lrelu', 
'identity'] # 'lrelu', 'lrelu', 'lrelu', 'lrelu', 'lrelu', 'identity']

assert len(layer_dim) ==  len(activations), f"layer dim or activation is missing.. act: {len(activations)}, dim: {len(layer_dim)}"

# hyper parameters of neural network
# for function 2 & 3
learning_rate = 8.8e-4
# for function 1
# learning_rate = 7e-3
num_epochs = 1500
mini_batch_size = 10


nn = NeuralNetwork(layer_dim, activations, learning_rate, num_epochs, mini_batch_size)

# train neural network 
nn.train(train_x, train_t, val_x, val_t, hah=True)


# test neural network 
train_loss, _ = nn.test(train_x, train_t)
print("training loss..", np.round(train_loss,4))
test_loss, test_output = nn.test(test_x, test_t)
print("testing loss..", np.round(test_loss,4))


# plot results 
fig = plt.gcf()
plt.plot(test_x.T, test_output.T, linewidth=3, color='red', label="output")
plt.plot(test_x.T, test_t.T, linewidth=3, color='blue', linestyle='dashed', label="target")
plt.title('function 3')
plt.legend()
plt.grid()
plt.show()
fig.savefig('function3_results try.png')