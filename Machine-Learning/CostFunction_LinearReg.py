# Linear Regression Cost Function:
# Here's how to obtain the co-efficents (theta0, theta1,...) in order to fit the best possible Linear Function
# to the training set. In other words, to reduce the sum of squared error (i.e. Error is the difference between the observed response in the dataset and the response predicted by the linear approximation)

from sklearn import linear_model

reg = linear_model.LinearRegression()

#`fit` method takes the arrays X and y as the input where X is the input array and y is the observed response.
#It stores the co-efficients of the linear model in `coef_`

reg.fit ([[0, 0], [1, 0], [2, 2]], [0, 1, 2])
#Output: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

reg.coef_
#Output: array([1., 0.])