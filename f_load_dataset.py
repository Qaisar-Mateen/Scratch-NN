import numpy as np

## create toy dataset for sin function
def create_toy_data(func, sample_size, std, domain=[-np.pi, np.pi]):
    x = np.linspace(domain[0], domain[1], sample_size)
    np.random.shuffle(x)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

def function1(x):
    return np.sin(2 * np.pi * x)*np.sin(x)

def function2(x):
    return np.sin(x)/x

def function3(x):
    return np.sin(x)*np.sin(2*x)


def load_dataset():   
    ### To generate figure 5 and 6 use this dataset
    
    domain_func0 = [0, 1]
    train_x, train_y = create_toy_data(function1, 1000, 0.5, domain_func0)
    val_x, val_y = train_x[900:,np.newaxis], train_y[900:,np.newaxis]
    train_x, train_y = train_x[0:900,np.newaxis], train_y[0:900,np.newaxis]  
    test_x = np.linspace(0, 1, 100)
    test_y = function1(test_x)    
    test_x, test_y = test_x[:,np.newaxis], test_y[:,np.newaxis]    
    
    ######################################################################################################################
    
    #### To generate figure 2, 3 and 4 use this dataset
    ## for figure 1 use function 1
    ## for figure 2 use function 2
    ## for figure 3 use function 3
    
    function = function1
    train_x, train_y = create_toy_data(function, 1000, 0.5)  
    val_x, val_y = train_x[900:,np.newaxis], train_y[900:,np.newaxis]
    train_x, train_y = train_x[0:900,np.newaxis], train_y[0:900,np.newaxis]
    test_x = np.linspace(-np.pi, np.pi, 100)
    test_y = function(test_x)    
    test_x, test_y = test_x[:,np.newaxis], test_y[:,np.newaxis]
    
    
##    print(train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape, test_y.shape)
    
    return train_x.T, train_y.T, val_x.T, val_y.T, test_x.T, test_y.T