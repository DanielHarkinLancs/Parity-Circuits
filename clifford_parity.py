import numpy as np
import scipy.sparse
import sympy as sp
import matplotlib as plt
import random
import stim
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys

sys.path.append(r'C:\Users\danie\Documents\Random Quantum Circuits\Amos efficient code')

from Lib_gen import *
from Lib_ed import *

def qSwap(q):
    A=np.zeros((q**2,q**2))
    A[q**2-1][q**2-1]=1
    
    for b in range(q):
        for a in range(q-1):
            if b>a:
                A[a*q+b][b*q+a]=1
                A[b*q+a][a*q+b]=1
            if a==b:
                A[a*q+b][b*q+a]=1
    return A

tabs = stim.Tableau.iter_all(2)
parity_sym=[]
s=qSwap(2)
k=0
for t in tabs:
    k+=1
    tab = t.to_unitary_matrix(endian='little')
    if np.allclose(s.dot(tab).dot(s),tab):
        parity_sym.append(t)
    else:
        continue

def local_parity_cliff(L,gates):
    layer1=stim.Tableau(0)
    for i in range(int(L/2)):
        layer1+=random.choice(gates)
    layer2=stim.Tableau(1)
    for i in range(int((L-1)/2)):
        layer2+=random.choice(gates)
    if L%2==0:
        layer2+=stim.Tableau(1)
    else:
        layer1+=stim.Tableau(1)
    return layer1.then(layer2)

def bigswapcliff(L):
    a=[]
    for b in range(L):
        p=stim.PauliString(L)
        p[L-1-b]=3
        a.append(p)
    return stim.Tableau.from_stabilizers(a)

def global_parity_cliff(L,sym_gates):
    left = stim.Tableau.random(int(L/2))
    left = left+stim.Tableau(int(L/2))
    swap = bigswapcliff(L)
    right = swap.then(left).then(swap)
    left = left.then(right)
    parity = stim.Tableau(int(L/2-1))+random.choice(sym_gates)+stim.Tableau(int(L/2-1))
    return left.then(parity)

def sff_cliff(u,L):
    pauli=stim.PauliString.iter_all(L)
    RHS=1
    for p in pauli:
        p_prime = u(p)
        if p_prime==p:
            RHS*=(1+p_prime.sign)
        elif p_prime==(-1)*p:
            return 0
    return RHS
