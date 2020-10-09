'''
    The Fixed Point Iteration
    Purpose:
        Computes approximate solution of f(x) = 0 by solving the fixed point
        problem x = g(x)
    Input: 
        g: function handle
        x0: initial guess
        tol: tolerance error
        iters: iteration number
        max_iters: maximum iteration number
    Output: 
        Approximate solution 

Last edited by Xiaozhou Li, September 15, 2018
'''

import numpy as np

def fixed_iters(g, x0, iters):
    x = np.zeros(iters+1)
    x[0] = x0
    for i in range(iters):
        x[i+1] = g(x[i])
    return x

def fixed_tol(g, x0, tol, max_iters=100):
    x_old = x0
    x_new = g(x_old)
    iters = 1
    while (np.abs(x_new - x_old) > tol):
        x_old = x_new
        x_new = g(x_old)
        iters += 1
        if (iters > 100):
            print ("Maximum iteration number achieved!")
            return
    return x_new


#########################################################
# Example
def g(x):
    return np.cos(x)

print (fixed_tol(g,0.5,1.e-12))
print (fixed_iters(g,0.5,20))
