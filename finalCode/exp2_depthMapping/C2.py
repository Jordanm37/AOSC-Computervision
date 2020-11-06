from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

'''
input = (left image x, left image y, right image x, right image y, real x, real y)
output = (z)
'''
# some random data (dimension similar to input)
X = np.random.rand(100,6)
Y = np.random.rand(100)
##X = [[0.44, 0.68, 0.34, 0.91, 0.56, 0.22], [0.99, 0.23, 0.68, 0.34, 0.91, 0.56]]
##Y = [109.85, 155.72]

# random unknown vector, will be used to predict
predict = [[0.44, 0.68, 0.34, 0.91, 0.56, 0.22]]

# generate polynomial of degree 4
poly = PolynomialFeatures(degree=4)
X_ = poly.fit_transform(X)

# fit polynomial
clf = linear_model.LinearRegression()
clf.fit(X_, Y)

# make prediction of unknown vector
predict_ = poly.fit_transform(predict)

# output
print (clf.predict(predict_))

# coeffs of polynomial
print(clf.coef_)
