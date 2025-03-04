import numpy as np
import matplotlib.pyplot as plt
from f_utils import *
import copy
from f_check_gradient import *
from sklearn.utils import shuffle


class NeuralNetwork():
  
    def __init__(self, num_neurons, activations_func, learning_rate, num_epochs, mini_batch_size):     
        self.num_neurons = num_neurons
        self.activations_func = activations_func
        self.learning_rate = learning_rate
        self.num_iterations = num_epochs
        self.mini_batch_size = mini_batch_size
        self.num_layers = len(self.num_neurons) - 1
        self.parameters = dict()
        self.net = dict()
        self.net1 = dict()
        self.grads = dict()

        
    def initialize_parameters(self):
        print("Num of layers", self.num_layers)
        for l in range(1, self.num_layers + 1):
            if self.activations_func[l] == 'relu': #xavier intialization method
                self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) / np.sqrt(self.num_neurons[l-1]/2.)
            else:                
                self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) / np.sqrt(self.num_neurons[l - 1])
            self.parameters['b%s' % l] = np.zeros((self.num_neurons[l], 1))
            print("weights and biases initialization",self.parameters['W%s'%l].shape,self.parameters['b%s'%l].shape)

          
    def fprop(self, batch_input): 
        self.net['A0'] = batch_input
        A_prev = batch_input
    
        for l in range(1, self.num_layers + 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
        
            # Linear transformation
            Z = np.dot(W, A_prev) + b
            self.net[f'Z{l}'] = Z
        
            # Activation
            if self.activations_func[l] == 'relu':
                A = relu(Z)
            elif self.activations_func[l] == 'sigmoid':
                A = sigmoid(Z)
            elif self.activations_func[l] == 'tanh':
                A = tanh(Z)
            elif self.activations_func[l] == 'lrelu':
                A = lrelu(Z, 0.01)
            elif self.activations_func[l] == 'identity':
                A = Z
            else:
                raise ValueError(f"Unsupported activation: '{self.activations_func[l]}'")
            
            self.net[f'A{l}'] = A
            A_prev = A
    
        self.net1['A%s' % self.num_layers] = A_prev
        return A_prev  
             
    def calculate_loss(self, batch_target):       
        AL = self.net1['A%s' % self.num_layers]
        m = batch_target.shape[1]
    
        # Using MSE loss
        loss = np.mean(np.square(AL - batch_target))
        return loss  
        
    def update_parameters(self,epoch):
        for l in range(1, self.num_layers + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * self.grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * self.grads[f'db{l}']
    
    def bprop(self, batch_target):
        m = batch_target.shape[1]
        L = self.num_layers
    
        # Initialize backprop
        dAL = 2 * (self.net1[f'A{L}'] - batch_target) / m
    
        for l in reversed(range(1, L + 1)):
            # Get cached values
            A_prev = self.net[f'A{l-1}']
            Z = self.net[f'Z{l}']
            W = self.parameters[f'W{l}']
            
            # Activation derivative
            if self.activations_func[l] == 'relu':
                dZ = relu_derivative(Z) * dAL
            elif self.activations_func[l] == 'sigmoid':
                dZ = sigmoid_derivative(self.net[f'A{l}']) * dAL
            elif self.activations_func[l] == 'tanh':
                dZ = tanh_derivative(self.net[f'A{l}']) * dAL
            elif self.activations_func[l] == 'lrelu':
                dZ = lrelu_derivative(Z, 0.01) * dAL
            elif self.activations_func[l] == 'identity':  # NEW CASE
                dZ = dAL.copy()
            else:
                raise ValueError(f"Unsupported activation: '{self.activations_func[l]}'")
        
            # Compute gradients
            self.grads[f'dW{l}'] = np.dot(dZ, A_prev.T)
            self.grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True)
        
            # Compute gradient for next layer
            if l > 1:
                dAL = np.dot(W.T, dZ)  
      
    
   

    def plot_loss(self,loss,val_loss):        
        plt.figure()
        fig = plt.gcf()
        plt.plot(loss, linewidth=3, label="train")
        plt.plot(val_loss, linewidth=3, label="val")
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title('learning rate =%s, hidden layers=%s' % (self.learning_rate, self.num_layers-1))
        plt.grid()
        plt.legend()
        plt.show()
        fig.savefig('plot_loss.png')
        
    
    def plot_gradients(self):
        avg_l_g = []
        grad = copy.deepcopy(self.grads)
        for l in range(1, self.num_layers+1):
#             print("layer %s"%l)
             weights_grad = grad['dW%s' % l]  
             dim = weights_grad.shape[0]
             avg_g = []
             for d in range(dim):
                 abs_g = np.abs(weights_grad[d])
                 avg_g.append(np.mean(abs_g))             
             temp = np.mean(avg_g)
             avg_l_g.append(temp)   
        layers = ['layer %s'%l for l in range(self.num_layers+1)]
        weights_grad_mag = avg_l_g
        fig = plt.gcf()
        plt.xticks(range(len(layers)), layers)
        plt.xlabel('layers')
        plt.ylabel('average gradients magnitude')
        plt.title('')
        plt.bar(range(len(weights_grad_mag)),weights_grad_mag, color='red', width=0.2) 
        plt.show() 
        fig.savefig('plot_gradients.png')
    

    def train(self, train_x, train_y, val_x, val_y):
        train_x, train_y = shuffle(train_x, train_y, random_state=0)
        self.initialize_parameters()        
        train_loss = []
        val_loss = []  
        num_samples = train_y.shape[1]       
        check_grad = True
        grad_ok = 0
        

        for i in range(0, self.num_iterations):
            for idx in range(0, num_samples, self.mini_batch_size):
                minibatch_input =  train_x[:, idx:idx + self.mini_batch_size]
                minibatch_target =  train_y[:, idx:idx + self.mini_batch_size]
                
                if check_grad == True:
                    self.fprop(minibatch_input) 
                    self.bprop(minibatch_target)
                    grad_ok = check_gradients(self, minibatch_input, minibatch_target)               
                    if grad_ok == 0:                           
                        print("gradients are not ok!\n")                           
                            
                   
                if grad_ok == 1:
                    check_grad = False
                    self.fprop(minibatch_input)
                    loss = self.calculate_loss(minibatch_target)
                    self.bprop(minibatch_target)           
                    self.update_parameters(i)
                   
            train_loss.append(loss) 
            self.fprop(val_x)
            va_loss = self.calculate_loss(val_y)
            val_loss.append(va_loss)              
            print("Epoch %i: training loss %f, validation loss %f" % (i, loss,va_loss))
        self.plot_loss(train_loss,val_loss)      
        self.plot_gradients()
       
    

   