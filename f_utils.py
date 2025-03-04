import numpy as np


def tanh(a):
    # return (np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a))  
    return np.tanh(a)

def tanh_derivative(a):
    return 1 - (np.tanh(a) ** 2) 

def relu(a):
    return np.maximum(0, a)
  
def sigmoid(a):
    return 1/(1 + np.exp(-a))

def relu_derivative(a): 
    return np.where(a > 0, 1, 0)

def sigmoid_derivative(a):
    # return np.exp(-a) / ((1 + np.exp(-a)) ** 2)
    s = sigmoid(a)
    return s * (1 - s)

def lrelu(a, k):
    # return a if a > 0 else k * a
    return np.where(a > 0, a, k * a)

def lrelu_derivative(a, k):  
    # return 1 if a > 0 else k
    return np.where(a > 0, 1, k)