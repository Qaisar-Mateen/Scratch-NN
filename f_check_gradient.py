import numpy as np
from f_utils import *
import copy



    
def check_gradients(self, train_X, train_t):
        ## add code here            
                
        eps= 1e-5
        grad_ok = 0
        
        for l in range(1, self.num_layers+1):  
                    ## add code here 
                    
                    diff = (np.linalg.norm(Numerical_grad - Analytical_grad))  / (np.linalg.norm(Numerical_grad) + np.linalg.norm(Analytical_grad))
                    print(Numerical_grad, Analytical_grad)
                              
                    if (diff> eps):
                        print("layer %s gradients are not ok"% l)  
                        grad_ok = 0
                    else:
                        print("layer %s gradients are ok"% l)
                        grad_ok = 1
              
        return grad_ok
         
            
        
        
        
        