import numpy as np
from f_utils import *
import copy


def check_gradients(self, train_X, train_t):
    eps = 1e-5 
    grad_ok = 1
    
    for l in range(1, self.num_layers + 1):
        # Get parameters and gradients for current layer
        W = self.parameters[f'W{l}']
        b = self.parameters[f'b{l}']
        dW = self.grads[f'dW{l}']
        db = self.grads[f'db{l}']
        
        # Initialize numerical gradients
        numerical_dW = np.zeros_like(W)
        numerical_db = np.zeros_like(b)
        
        # --- Compute numerical gradients for weights ---
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                original = W[i, j].copy()
                
                # Positive perturbation
                W[i, j] = original + eps
                self.fprop(train_X)
                loss_plus = self.calculate_loss(train_t)
                
                # Negative perturbation
                W[i, j] = original - eps
                self.fprop(train_X)
                loss_minus = self.calculate_loss(train_t)
                
                # Numerical gradient
                numerical_dW[i, j] = (loss_plus - loss_minus) / (2 * eps)
                
                # Restore original value
                W[i, j] = original
        
        # --- Compute numerical gradients for biases ---
        for i in range(b.shape[0]):
            original = b[i].copy()
            
            # Positive perturbation
            b[i] = original + eps
            self.fprop(train_X)
            loss_plus = self.calculate_loss(train_t)
            
            # Negative perturbation
            b[i] = original - eps
            self.fprop(train_X)
            loss_minus = self.calculate_loss(train_t)
            
            # Numerical gradient
            numerical_db[i] = (loss_plus - loss_minus) / (2 * eps)
            
            # Restore original value
            b[i] = original
        
        # Combine gradients into vectors
        Numerical_grad = np.concatenate([numerical_dW.flatten(), numerical_db.flatten()])
        Analytical_grad = np.concatenate([dW.flatten(), db.flatten()])
        
        # Calculate relative difference
        numerator = np.linalg.norm(Numerical_grad - Analytical_grad)
        denominator = np.linalg.norm(Numerical_grad) + np.linalg.norm(Analytical_grad)
        diff = numerator / denominator
        
        # Debug prints (optional)
        # print(f"Layer {l} numerical grad:", Numerical_grad[:5])
        # print(f"Layer {l} analytical grad:", Analytical_grad[:5])
        
        # Check gradient validity
        if diff > eps:
            print(f"Layer {l} gradients are problematic (diff: {diff:.2e})")
            grad_ok = 0
        else:
            print(f"Layer {l} gradients are OK (diff: {diff:.2e})")
    
    return grad_ok
    
# def check_gradients(self, train_X, train_t):
#         ## add code here            
                
#         eps= 1e-5
#         grad_ok = 0
        
#         for l in range(1, self.num_layers+1):  
#                     ## add code here 
                    
#                     diff = (np.linalg.norm(Numerical_grad - Analytical_grad))  / (np.linalg.norm(Numerical_grad) + np.linalg.norm(Analytical_grad))
#                     print(Numerical_grad, Analytical_grad)
                              
#                     if (diff> eps):
#                         print("layer %s gradients are not ok"% l)  
#                         grad_ok = 0
#                     else:
#                         print("layer %s gradients are ok"% l)
#                         grad_ok = 1
              
#         return grad_ok