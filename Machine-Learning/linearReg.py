import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

#dimensions and sample:
print(diabetes.data.shape)
print(diabetes.data[:10])

#Using one feature:
data_X = diabetes.data[:, np.newaxis, 2]
print(data_X.data.shape)

#Split data into Training and Test sets.
#'data’: the data to learn 
#‘target’: the regression target for each sample.
data_X_train = data_X[:-20]
data_X_test = data_X[-20:]
data_y_train = diabetes.target[:-20]
data_y_test = diabetes.target[-20:]

#Creating linear regression object:
# LinerRegression takes three parameters:

linearRegr = linear_model.LinearRegression()

#Training the model using the Training sets:
linearRegr.fit(data_X_train,data_y_train)

#Predict the test set:
prediction_y = linearRegr.predict(data_X_test)

print('Co-efficients: ', linearRegr.coef_)
print("Mean Square Root Error: %.2f" % mean_squared_error(data_y_test,prediction_y))
print("Variance score: %.2f" % r2_score(data_y_test,prediction_y))

#Plotting the output
plt.scatter(data_X_test, data_y_test, color='black')
plt.plot(data_X_test,prediction_y, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()