#2 layer neural network
import numpy as np
import time

#variables
n_hidden = 10 # number of hidden neurons, array of 10 input values and compare to 10 other values and compute XOR
n_in = 10 #outputs
n_out = 10
n_samples = 300

#hyperparameters
learning_rate = 0.01 #defines how fast we want to netowrk to learn
momentum = 0.9

np.random.seed(0) #seed ensures that we will generate the same "random" values every time we run the code

#activation function -
#sigmoid function - turns numbers into probabilities
#input data which is numbers when come through neural, each of weight is a set of probabilities
#this probabilities are updated when we train out network
#every time input data hits one of neurons it is going to turn number into probability

#we will use 2 activation functions
def sigmoid(x): #for first layer
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x): #for second layer
    return 1 - np.tanh(x) ** 2

#train function
#x - input data
#t - transpose? will help make multiplication
#V, W - layers of out network
#bv, bw - biases - will help make better prediction, one bias for one layer in network
#input data, transpose, layer 1, layer 2, biases
def train(x, t, V, W, bv, bw):
    #forward propagation - matrix multiply + biases
    #we are taking dot product of input data x and we are putting it into out first layer V, A is a delta value
    A = np.dot(x, V) + bv
    Z = np.tanh(A) #perform activation function on our data

    B = np.dot(Z, W) + bw # putting into 2 layer
    Y = sigmoid(B)

    #backward propagarion
    #t - matrix of out weights filped, we want the filped version to go backwards
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)
    #Ev is used to predict out loss, to minimize loss, that's how we train

    #predict loss
    dW = np.outer(Z, Ew) #Z value, that we predicted from tanh
    dV = np.outer(x, Ev) #x - input
    #dW, dV - deltas to calculate loss

    #cross entropy, becouse we are doing classification
    loss = -np.mean(t * np.log(Y) + (1 -t) * np.log(1-Y))

    return loss, (dV, dW, Ev, Ew)

def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)

#create layers
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V, W, bv, bw]

#generate data
X = np.random.binomial(1, 0.5, (n_samples, n_in))
T = X ^ 1

#Training time

for epoch in range(100):
    err = []
    upd = [0] * len(params)

    t0 = time.clock()
    #for each data point we want to update weights of out network
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i],  *params)
        #update loss
        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum + upd[j]

        err.append(loss)

    #print('Epoch %d, Loss: %.8f, Time: %.4f s' %( epoch, np.mean(err), time.clock()-t0))
    print("Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 ))
#try to predict sth
x = np.random.binomial(1, 0.5, n_in)
print ('XOR Predict')
print (x)
print(predict(x, *params))



























