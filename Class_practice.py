#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kemistree4
"""


import numpy as np
import matplotlib.pylab as plt

#Generate and sort random data points

x = np.array([np.random.randint(1, 50, 50)])
y = np.array([np.random.randint(1, 50, 50)])
x.sort()
y.sort()
x = x[0]
y = y[0]
plt.scatter(x, y)

#Compute slope
m = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x) ** 2)

#Compute bias
b = (np.sum(y) - m * np.sum(x)) / len(x)

print(m)
print(b)

def predict(x,m=0,b=0):
    return m * (x) + b


def fit(x,y):
 
    #def inner(x1):
    #    return m * x1 + b
    
    m = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x) * np.sum(x))
    b = (np.sum(y) - m *np.sum(x)) / len(x)
    return dict(m=m, b=b)

model_dict_1 = fit(x,y)
#y_prediction = predict(x, m=model['m'], b=model['b']) 
y_prediction = predict(x, **model_dict_1)  
#plt.plot(x,y_prediction)


class UglyLinearRegression:
    
    def predict(self, x, m=0, b=0):
        return m * (x) + b

    def fit(self, x, y):
 
        #def inner(x1):
        #    return m * x1 + b
        
        m = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x) * np.sum(x))
        b = (np.sum(y) - m *np.sum(x)) / len(x)
        return dict(m=m, b=b)
    
model = UglyLinearRegression()

model_dict = model.fit(x, y)
line_pred = model.predict(x, m=model_dict['m'], b=model_dict['b'] )
plt.plot(x, line_pred)


class PlainLinearRegression:
    
    def predict(self, x):
        return self.m * (x) + self.b

    def fit(self, x, y):
 
        #def inner(x1):
        #    return m * x1 + b
        
        self.m = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x) * np.sum(x))
        self.b = (np.sum(y) - m *np.sum(x)) / len(x)
        return self
    
model = PlainLinearRegression()

model = model.fit(x, y)
line_pred = model.predict(x)
plt.plot(x, line_pred)


class RegularizedLinearRegression:
    
    def __init__(self, alpha=0):
        self.alpha = alpha    
        
    def predict(self, x):
        return self.m * (x) + self.b

    def fit(self, x, y):
 
        #def inner(x1):
        #    return m * x1 + b
        self.m, self.b = 0, 0
        for i in range(10):    
            self.m = (len(x) * np.sum(x*y) - np.sum(x) * (np.sum(y))+ np.abs(self.m + self.b) * self.alpha) / (len(x)*np.sum(x*x) - np.sum(x) * np.sum(x))
            self.b = (np.sum(y) - m *np.sum(x)) / len(x)
        return self
    
model = RegularizedLinearRegression(alpha=10)

model = model.fit(x, y)
line_pred = model.predict(x)
plt.plot(x, line_pred)