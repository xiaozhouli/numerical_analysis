'''
    Newton's Method
    Purpose:
        Computes LU factorization of matrix A (without pivoting)
    Input: 
        A: the n x n matrix
    Output: 
        A: the compact form of the LU factorization of A 

Last edited by Xiaozhou Li, September 23, 2020
'''

import numpy as np

def LU_factor(A):
    A = A.astype(np.float)
    n = len(A)
    for j in range(0,n-1):
        if A[j,j] != 0:
            for i in range(j+1,n):
                lam = A[i,j]/A[j,j]
                A[i,j+1:n] = A[i,j+1:n] - lam*A[j,j+1:n]
                A[i,j] = lam
    return A

#########################################################
# Example
A = np.array([[2, 4, 4, 2], [3, 3, 12, 6], [2, 4, -1, 2], [4, 2, 1, 1]])
print ("The matrix A: \n", A)

print ("The LU factorization of A: \n", LU_factor(A))
