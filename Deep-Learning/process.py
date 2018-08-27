import numpy as np
import pandas as pd

def get_data():
	df = pd.read_csv('ecommerce_data.csv')

	data = df.as_matrix()  #converts the data into matrix to be able to work with numpy

	X = data[:,:-1]        #Contains all columns except last column - user_action
	Y = data[:,-1]		   #An array of only the user_action	

	X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()  #processing the second col - no. of products viewed
	X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()  #processing the third col - visit duration

	N,D = X.shape
	#new array with the same number of cols but three more extra rows in order to process the col - time_of_day
	X2 = np.zeros((N,D+3))	

	#copying data from X for all the other cols (except last three)
	X2[:,0:(D-1)] = X[:,0:(D-1)]	

	#populating the last three cols based on time_of_day (Ex: If time_of_day is 3, the last col set to True)
	for n in range(N):				
		t = int(X[n,D-1])		
		X2[n,t+D-1] = 1			

	return X2,Y	

def get_binary_data():
	X, Y = get_data()
	X2 = X[Y <= 1]
	Y2 = Y[Y <= 1]
	return X2,Y2
