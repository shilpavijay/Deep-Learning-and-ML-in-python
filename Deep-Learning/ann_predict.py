import numpy as np
from process import get_data

X, Y = get_data()

#Initializing random weights for the Neural Network:
 
D = X.shape[1]   #No.of columns in X - input layer
M = 5			 #Layer1 (hidden layer) of the NN (assumption made)  	
K = len(set(Y))  #output layer of the NN - Types of outputs possible (=set(user_actions) : 0/1/2/3 )

W1 = np.random.randn(D,M)   #Array of Weights from i/p to the hidden layer
b1 = np.zeros(M)		#Bias on the hidden layer
W2 = np.random.randn(M,K)   #Weights from Hidden to o/p layer
b2 = np.zeros(K)		#bias on the output layer

def softmax(arr):
	expA = np.exp(arr)
	return expA / expA.sum(axis=1, keepdims=True)

def forward(W1,b1,W2,b2):
	Z = np.tanh(X.dot(W1) + b1)
	return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(W1,b1,W2,b2)
prediction = np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y,P):
	return np.mean(Y == P)

print("Score: ", classification_rate(Y,prediction))