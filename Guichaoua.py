#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guichaoua
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import io
from cvxopt import matrix, solvers

###############################################################################
def base4(chaine, k):
    """ avec chaine de longueur k"""
    assert len(chaine)==k, "erreur dans la longueur de chaine"
    return np.sum([alphabet.index(chaine[p])*4**(k-1-p) for p in range(k)])

def transforme(X,k):
    Y = np.array([[base4(X[j][i:i+k], k) for i in range(len(X[j])-k+1)] for j in range(len(X))])
    return np.array([ np.histogram(Y[i,:], bins = np.arange(4**k+1)-0.5)[0] for i in range(len(Y))])

def base4_m(chaine, k):
    """ avec chaine de longueur k"""
    assert len(chaine)==k, "erreur dans la longueur de chaine"
    T = []
    for i in range(k):
        S1 = np.sum([ alphabet.index(chaine[p])*4**(k-1-p) for p in range(0,i)])
        S2= np.sum([ alphabet.index(chaine[p])*4**(k-1-p) for p in range(i+1,k)])
        for j in range(4):
            T.append(S1+S2+j*4**(k-1-i))
    return T

def transforme_m(X,k):
    Y = np.array([[base4_m(X[j][i:i+k], k) for i in range(len(X[j])-k+1)] for j in range(len(X))])
    a,b,c = Y.shape
    Z = Y.reshape(a,b*c)
    return np.array([ np.histogram(Z[i,:], bins = np.arange(4**k+1)-0.5)[0] for i in range(len(Z))])

###############################################################################
alphabet = ['A','C','G','T']
T = np.array([[0]*1000 for i in range(3)])
L = [0.001, 0.003, 0.001]
k = 9
for num in range(3):
    lamda = L[num]

    f = open('Xtr'+str(num)+'.csv', "r")
    lignes = f.readlines()
    table = [ligne.rstrip().split(",") for ligne in lignes]
    X = np.array(table[1:])
    X = np.array(X[:,1])
    f.close()

    X_0 = transforme(X,k)
    X_1 = transforme_m(X,k)
    X_11 = X_1 - X_0*(k-1) # le "vrai" substring kernel
    X_tr = X_11 / np.mean( np.sqrt(np.sum(X_11**2,axis=1)) )# approximativement normalisé en norme L2

    y_tr=np.genfromtxt('Ytr'+str(num)+'.csv',dtype=float,delimiter=',',skip_header=1)[:,1]
    y_tr = 2*y_tr-1 # pour avoir -1; 1

    f = open('Xte'+str(num)+'.csv', "r")
    lignes = f.readlines()
    table = [ligne.rstrip().split(",") for ligne in lignes]
    X = np.array(table[1:])
    X = np.array(X[:,1])
    f.close()
    X_0 = transforme(X,k)
    X_1 = transforme_m(X,k)
    X_11 = X_1 - X_0*(k-1) # le "vrai" substring kernel
    X_val = X_11 / np.mean( np.sqrt(np.sum(X_11**2,axis=1)) )# approximativement normalisé en norme

    n,d = X_tr.shape
    ny = len(y_tr)
    w = X_tr.T @ np.linalg.solve(X_tr @ X_tr.T + lamda*ny*np.eye(n),y_tr)
    T[num,:] =  (np.sign(X_val@w)+1)//2

Q = np.reshape(T,3000)
Q = [[i,Q[i]] for i in range(3000)]
np.savetxt('GUICHAOUA_k10.csv', Q , delimiter=',', fmt = '%i',header = 'Id,Bound',comments='')
