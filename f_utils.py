import numpy as np


def tanh(a):
    return (np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a))  

def tanh_derivative(a):
    return 1 - (tanh(a) ** 2) 

def relu(a):
    return a if a > 0 else 0
  
def sigmoid(a):
    return 1/(1 + np.exp(-a))

def relu_derivative(a): 
    return 1 if a > 0 else 0

def sigmoid_derivative(a):
    # return np.exp(-a) / ((1 + np.exp(-a)) ** 2)
    return sigmoid(a) * (1 - sigmoid(a))

def lrelu(a, k):
    return a if a > 0 else k * a

def lrelu_derivative(a, k):  
    return 1 if a > 0 else k





