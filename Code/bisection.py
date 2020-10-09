'''
    Bisection Method
    Purpose:
        Computes approximate solution of f(x) = 0
    Input: 
        f: function handle
        a, b: interval [a, b], such that f(a)*f(b)<0 and f(x) = 0 has a unique
        solution over [a, b]
        tol: tolerance error
    Output: 
        Approximate solution 

Last edited by Xiaozhou Li, September 9, 2018
'''


import numpy as np
import matplotlib.pyplot as plt

def bisect(f, a, b, tol):
    if (np.sign(f(a)*f(b)) >= 0):      # wrong input
        print ('$f(a)f(b) <0$ not satisfied')
        return
    fa = f(a) 
    fb = f(b)
    while ((b - a)/2) > tol:
        c = (a + b)/2;
        print (c)
        fc = f(c);
        if fc == 0:                 # c is a solution, done
            return c 
        if (np.sign(fc*fa) < 0):       # new interval [a, c]
            b = c; 
            fb = fc
        else:                       # new interval [c, b]
            a = c
            fa = fc
    return (a + b)/2                #new midpoint is best estimate

#########################################################
# Example
def fun(x):
    return np.exp(-x) - np.sin(np.pi/2*x)

print (bisect(fun,0,1,1.e-12))
