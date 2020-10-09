'''
    Newton's Method
    Purpose:
        Computes approximate solution of f(x) = 0 by Newton's method
    Input: 
        f: function handle
        f_der: the derivative of function f
        x0: initial guess
        tol: tolerance error
        iters: iteration number
        max_iters: maximum iteration number
    Output: 
        Approximate solution 

Last edited by Xiaozhou Li, September 23, 2020
'''

import numpy as np

def newton_iters(f, f_der, x0, iters):
    x = np.zeros(iters+1)
    x[0] = x0
    for i in range(iters):
        x[i+1] = x[i] - f(x[i])/f_der(x[i])
    return x

def newton_tol(f, f_der, x0, tol, max_iters=100):
    x_old = x0
    x_new = x_old - f(x_old)/f_der(x_old)
    iters = 1
    while (np.abs(x_new - x_old) > tol):
        x_old = x_new
        x_new = x_old - f(x_old)/f_der(x_old)
        iters += 1
        if (iters > 100):
            print ("Maximum iteration number achieved!")
            return
    return x_new

#########################################################
# Example
def f(x):
    return x**3 + 10*x - 20
    
def f_der(x):
    return 3*x**2 + 10

print (newton_tol(f, f_der, 1.5, 1.e-12))
print (newton_iters(f, f_der, 1.5, 20))
