import numpy as np

X= np.array([[1,2]])

D = 2			 #No.of columns in X - input layer
M = 1			 #Layer1 (hidden layer) of the NN (assumption made)  	
K = 2 			 #output layer of the NN - Types of outputs possible (=set(user_actions) : 0/1/2/3 )

W1 = np.array([[1,1],[1,0]])  #Array of Weights from i/p to the hidden layer
b1 = 0						  #Bias on the hidden layer
W2 = np.array([[0,1],[1,1]])  #Weights from Hidden to o/p layer
b2 = 0						  #bias on the output layer

def softmax(arr):
	expA = np.exp(arr)
	return expA / expA.sum(axis=1, keepdims=True)

def forward(W1,b1,W2,b2):
	Z = np.tanh(X.dot(W1) + b1)
	return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(W1,b1,W2,b2)

print(P_Y_given_X)
