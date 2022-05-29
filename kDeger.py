# -*- coding: utf-8 -*-
"""
Created on Sun May 29 02:54:02 2022

@author: bolat
"""

import matplotlib.pyplot  as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def ka(s):
    x = np.array([0.4,0.6,0.8,1.0,1.2,1.4,1.6]).reshape((-1,1))
    y = np.array([1,0.74,0.7,0.67,0.65,0.63,0.62])
    

    
    poly = PolynomialFeatures(degree = 6)
    x_poly = poly.fit_transform(x)
    
    reg = LinearRegression()
    
    lin = reg.fit(x_poly,y)
    
    
    pre_x = np.linspace(0,1.6,100).reshape(-1,1)
    pre_y = lin.predict(poly.fit_transform(pre_x))
    
    su = np.array(s).reshape(-1,1)
    
    ans_x = lin.predict(poly.fit_transform(su))
    
    plt.scatter(x,y)
    plt.scatter(su,ans_x,color="red" ,label ="istenilen ka degeri")
    plt.plot(pre_x,pre_y)
    plt.xlabel("ultimate strenght GPa")
    plt.xlim(0,1.7)
    plt.ylim(0.5, 1)
    plt.ylabel("Surface Factor ka")
    plt.legend()
    
    plt.show()
    
    return round(float(ans_x),3)


def kb(m):
    
    M = np.array([1,2,2.25,2.5,2.75,3,3.5,4,4.5,5,5.5,6,7,8,9,
                  10,11,12,14,16,18,20,22,25,28,32,36,40,45,50]).reshape(-1,1)
    kB = np.array([1,1,0.984,0.974,0.965,0.956,0.942,0.930,0.920,
                   0.910,0.902,0.894,0.881,0.870,0.860,0.851,0.843,
                  0.836,0.824,0.813,0.804,0.796,0.788,0.779,0.770,
                  0.760,0.752,0.744,0.736,0.728])
    
    poly = PolynomialFeatures(degree = 6)
    x_poly = poly.fit_transform(M)
    
    reg = LinearRegression()
    
    lin = reg.fit(x_poly,kB)
    
    
    pre_x = np.linspace(0,50,1000).reshape(-1,1)
    pre_y = lin.predict(poly.fit_transform(pre_x))
    
    m = np.array(m).reshape(-1,1)
    
    ans_x = lin.predict(poly.fit_transform(m))
    
    plt.scatter(M,kB,s=1)
    plt.scatter(m,ans_x,color="red" ,label ="istenilen kb degeri")
    plt.plot(pre_x,pre_y)
    plt.xlabel("module")
    plt.ylabel("Size factor kb")
    plt.legend()
    
    plt.show()
    
    return round(float(ans_x),3)


def kc(r):
    
    R  =pd.DataFrame(data= {"reliability":[0.5,0.9,0.95,0.99,0.999,0.9999],
       "Factor kc":[1.0,0.897,0.868,0.814,0.753,0.702]})
    
    for i in range(0,len(R["reliability"])):
        
        if R["reliability"][i] ==  r:
            return R["Factor kc"][i]
        
        
def kd(t=350):
    if t<=350:
        return 1
    elif t>350 or t<500:
        return 0.5
    
        
        
    
    

   
   
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    