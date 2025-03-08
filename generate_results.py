import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from f_load_dataset import load_dataset

########################################################################################
# learning_rate = 1e-3
# epochs = 100
# hidden layers = 5
# num of neurons = 100
# mini_batch_size = 25

# activation functions vs errors
def generate_fig6():
    train_x, train_t, val_x, val_t, test_x, test_t = load_dataset()

    # Parameters
    learning_rate = 1e-3
    epochs = 100
    mini_batch_size = 25

    train_loss = []
    test_loss = []
    width = 0.3
    act_func = ['sigmoid', 'tanh', 'relu', 'lrelu']

    for func in act_func:
        layer_dim = [1, 100, 100, 100, 100, 100, 1]
        activations = [None, func, func, func, func, func, 'identity']
        nn = NeuralNetwork(layer_dim, activations, learning_rate, epochs, mini_batch_size)
        nn.train(train_x, train_t, val_x, val_t)

        a, _ = nn.test(train_x, train_t)
        train_loss.append(a)

        b, _ = nn.test(test_x, test_t)
        test_loss.append(b)

    fig = plt.gcf()
    plt.xticks(range(len(act_func)), act_func)
    plt.xlabel('activation functoins')
    plt.ylabel('loss')
    plt.bar(np.arange(len(train_loss)),train_loss, width=width, label='train loss') 
    plt.bar(np.arange(len(test_loss))+width,test_loss, width=width, label='test loss') 
    plt.legend()
    plt.show()
    fig.savefig('Figure_6', dpi=300)


########################################################################################
# activation functions vs hidden layers
# learning_rate = 1e-3
# epochs = 100
# num of neurons = 50
# mini_batch_size = 10

def generate_fig5():
    test_error_sigmoid = [0.05872, 0.06902, 0.09763, 0.10762, 0.10923]
    test_error_tanh = [0.056213, 0.044345, 0.03815, 0.04221, 0.02421]
    test_error_relu = [0.05472, 0.04523, 0.03521, 0.020109, 0.021035]
    test_error_lrelu = [0.054234, 0.04698, 0.03356, 0.031901, 0.02123] 
    act_func = ['1HL', '2HL', '3HL', '5HL', '10HL']

    # Load dataset
    train_x, train_t, val_x, val_t, test_x, test_t = load_dataset()

    hidden_layer_counts = [1, 2, 3, 5, 10]

    # Parameters
    learning_rate = 1e-3
    epochs = 100
    num_neurons_per_hidden = 50
    mini_batch_size = 10

    activation_funcs = ['sigmoid', 'tanh', 'relu', 'lrelu']

    for hl in hidden_layer_counts:
        for func in activation_funcs:

            layer_dim = [1] + [num_neurons_per_hidden] * hl + [1]

            activations = [None] + [func] * hl + ['identity']
            nn = NeuralNetwork(layer_dim, activations, learning_rate, epochs, mini_batch_size)

            nn.train(train_x, train_t, val_x, val_t)

            error, _ = nn.test(test_x, test_t)
            if func == 'sigmoid':
                test_error_sigmoid.append(error)
            elif func == 'tanh':
                test_error_tanh.append(error)
            elif func == 'relu':
                test_error_relu.append(error)
            elif func == 'lrelu':
                test_error_lrelu.append(error)

            print(f'Activation: {func}, Hidden Layers: {hl}, Test Loss: {error}')

    fig = plt.gcf()
    plt.xticks(range(len(act_func)), act_func)
    plt.xlabel('hidden layers')
    plt.ylabel('loss')
    plt.plot(test_error_sigmoid, color='darkorange',linestyle='dashed',linewidth=2, marker='o', markerfacecolor='darkorange', markersize=8, label='sigmoid') 
    plt.plot(test_error_tanh, color='red',linestyle='dashed',linewidth =2, marker='o', markerfacecolor='red', markersize=8, label='tanh') 
    plt.plot(test_error_relu, color='purple',linestyle='dashed',linewidth =2, marker='o', markerfacecolor='purple', markersize=8, label='relu') 
    plt.plot(test_error_lrelu, color='blue',linestyle='dashed',linewidth =2, marker='o', markerfacecolor='blue', markersize=8, label='lrelu') 
    plt.legend()
    plt.show()
    fig.savefig('Figure_5', dpi=300)

########################################################################################

if __name__ == '__main__':
    generate_fig5()
    generate_fig6()