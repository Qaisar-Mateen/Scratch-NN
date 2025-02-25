import numpy as np
import matplotlib.pyplot as plt

########################################################################################
# learning_rate = 1e-3
# epochs = 100
# hidden layers = 5
# num of neurons = 100
# mini_batch_size = 25
# activation functions vs errors
train_loss = []
test_loss = []
width = 0.3
act_func = ['sigmoid', 'tanh', 'relu', 'lrelu']
fig = plt.gcf()
plt.xticks(range(len(act_func)), act_func)
plt.xlabel('activation functoins')
plt.ylabel('loss')
plt.bar(np.arange(len(train_loss)),train_loss, width=width, label='train loss') 
plt.bar(np.arange(len(test_loss))+width,test_loss, width=width, label='test loss') 
plt.legend()
plt.show()
fig.savefig('Figure_4', dpi=300)


########################################################################################
# activation functions vs hidden layers
# learning_rate = 1e-3
# epochs = 100
# num of neurons = 50
# mini_batch_size = 10
test_error_sigmoid = []
test_error_tanh = []
test_error_relu = []
test_error_lrelu = []
act_func = ['1HL', '2HL', '3HL', '5HL', '10HL']
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








