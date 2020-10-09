'''
    Iterative Method
    Purpose:
        Computes approximate solution of Ax = b by the classic iterative method
    Including methods:
        - Jacobi Method
        - Gauss-Seidel Method
        - Successive Over-Relaxation (SOR) Method
        - Conjugate Gradient (CG) Method 
    Input: 
        A: the n x n matrix
        b: the right hand vector
        x0: initial guess
        tol: tolerance error
    Output: 
        x: the approximate solution
        iters: iteration number

Last edited by Xiaozhou Li, September 23, 2020
'''

import numpy as np

def Jacobi_tol(A, b, x0, tol):
    x_old = np.copy(x0)
    x_new = np.zeros(np.size(x0))
    for i in range(np.size(x0)): 
        x_new[i] = (b[i] - np.dot(A[i,:i],x_old[:i]) - np.dot(A[i,i+1:],x_old[i+1:]))/A[i,i]
    iters = 1
    while ((np.linalg.norm(x_new-x_old,np.inf)) > tol):
        x_old = np.copy(x_new)
        for i in range(np.size(x0)): 
            x_new[i] = (b[i] - np.dot(A[i,:i],x_old[:i]) - np.dot(A[i,i+1:],x_old[i+1:]))/A[i,i]
        iters += 1
    return x_new, iters


def GS_tol(A, b, x0, tol):
    x_old = np.copy(x0)
    x = np.copy(x0)
    for i in range(np.size(x0)): 
        x[i] = (b[i] - np.dot(A[i,:i],x[:i]) - np.dot(A[i,i+1:],x[i+1:]))/A[i,i]
    iters = 1
    while ((np.linalg.norm(x-x_old,np.inf)) > tol):
        x_old = np.copy(x)
        for i in range(np.size(x0)): 
            x[i] = (b[i] - np.dot(A[i,:i],x[:i]) - np.dot(A[i,i+1:],x[i+1:]))/A[i,i]
        iters += 1
    return x, iters


def SOR_tol(A, b, x0, omega, tol):
    x_old = np.copy(x0)
    x = np.copy(x0)
    for i in range(np.size(x0)): 
            x[i] = x[i] + omega*(b[i] - np.dot(A[i,:i],x[:i]) - np.dot(A[i,i:],x[i:]))/A[i,i]
    iters = 1
    while ((np.linalg.norm(x-x_old,np.inf)) > tol):
        x_old = np.copy(x)
        for i in range(np.size(x0)): 
            x[i] = x[i] + omega*(b[i] - np.dot(A[i,:i],x[:i]) - np.dot(A[i,i:],x[i:]))/A[i,i]
        iters += 1
    return x, iters

def CG_tol(A, b, x0, x_star, tol):
    r_new = b - np.dot(A, x0) 
    r_old = np.copy(np.size(x0))
    d_old = np.zeros(np.size(x0))
    x = np.copy(x0)
    iters = 0
    while ((np.linalg.norm(x-x_star,np.inf)) > tol):
        if (iters == 0):
            d_new = np.copy(r_new)
        else:
            beta = np.dot(r_new,r_new)/np.dot(r_old,r_old)
            d_new = r_new + beta*d_old
        Ad = np.dot(A, d_new)
        alpha = np.dot(r_new,r_new)/np.dot(d_new,Ad)
        x += alpha*d_new
        d_old = d_new
        r_old = r_new
        r_new = r_old - alpha*Ad
        iters += 1
    return x, iters

#########################################################
# Example
def Iterative_solver(n, tol=1.e-8):
    # Create matrix A with size n x n
    A = 3*np.eye(n) - np.diag(np.ones(n-1),-1) - np.diag(np.ones(n-1),+1)
    for i in range(n):
        if (abs(n-1 - 2*i) > 1):
            A[i, n-1-i] = - 1/2
    
    if (n <= 10):
        print ("The Matrix A: \n", A)
    # Exact solution v = [1, 1, ..., 1]^T
    v = np.ones(n)
    b = np.dot(A, v)

    # initial guess
    v0 = np.zeros(np.size(b))

    v_J, iters = Jacobi_tol(A, b, v0, tol)
    print ("Jacobi Method:  %4d    %7.2e" %(iters, np.linalg.norm(v - v_J, np.inf)))
    
    v_GS, iters = GS_tol(A, b, v0, tol)
    print ("Gauss Seidel :  %4d    %7.2e" %(iters, np.linalg.norm(v - v_GS, np.inf)))
    
    omega = 1.25
    v_SOR, iters = SOR_tol(A, b, v0, omega, tol)
    print ("SOR Method   :  %4d    %7.2e" %(iters, np.linalg.norm(v - v_SOR, np.inf)))
    
    #v_CG, iters = CG_tol(A, b, v0, v, tol)
    #print ("CG Method    :  %4d    %7.2e" %(iters, np.linalg.norm(v - v_CG, np.inf)))

Iterative_solver(10, 1.e-8)
