import numpy as np
import scipy as sp
import sys
import os

sys.path.append(r'/mmfs1/home/users/harkind/Random Circuits')

from Lib_gen import *
from Lib_ed import *

os.environ['OMP_NUM_THREADS'] = '1'

L =int(sys.argv[3])        ## Length of the tight binding chain
q=int(sys.argv[1])
eps=float(sys.argv[2])
qt=int(sys.argv[4])

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

def parity_gate(q,eps):
    Q = qSwap(q)
    
    A = np.random.normal(0,eps/4,(q**2,q**2))

    B = np.random.normal(0,eps/4,(q**2,q**2))
    
    h = A+1j*B
    
    h = h + (h.conj().T)
    
    gamma = h+(Q.dot(h)).dot(Q)
    
    return sp.linalg.expm(1j*gamma)

def parity_cue(q,L,eps,list=True):
    s = big_swap_matrix(q,L)
    left = W_cue_brickwall(q,int(L/2),True,False)  
    left = [sp.sparse.kron(a,sp.sparse.identity(int(q**(L/2)))) for a in left]
    right = [(s.dot(a)).dot(s) for a in left]
    parity = Sparse2gate_general_multiarg(parity_gate,q,L,int(L/2),q,eps)
    #Test
    W = []
    
    W += left
    W += right
    W.append(parity)
    
    if list:
        return W
    
    else:
        temp = W[0]
        for (i,mat) in enumerate(W):
            if not(i==0):
                temp = mat.dot(temp)
                
            else:
                pass
        return temp.toarray()

sff = SFF_Double_Stochastic_Lin_Op(q, L,200,parity_cue,False,q,L,eps,True)

A_even, A_odd, even_parity_basis_set, odd_parity_basis_set = block_diag_parity(q,L)
A = np.vstack((A_even,A_odd))
A_inv =np.hstack((A_even.T,A_odd.T))

circuit=parity_cue(q,L,eps,False)

E=np.linalg.eigvals(A_even.dot(circuit.dot(np.transpose(A_even))))
ph_array=np.sort(np.angle(E))

avg_ratio=0
for j in range(len(E)-2):
    d1=ph_array[j+1]-ph_array[j]
    d2=ph_array[j+2]-ph_array[j+1]
    avg_ratio=avg_ratio+abs(min(d1,d2)/max(d1,d2))
avg_ratio_even=avg_ratio/(len(E)-2)

E=np.linalg.eigvals(A_odd.dot(circuit.dot(np.transpose(A_odd))))
ph_array=np.sort(np.angle(E))

avg_ratio=0
for j in range(len(E)-2):
    d1=ph_array[j+1]-ph_array[j]
    d2=ph_array[j+2]-ph_array[j+1]
    avg_ratio=avg_ratio+abs(min(d1,d2)/max(d1,d2))
if len(E)==2:
    avg_ratio_odd=0
else:
    avg_ratio_odd=avg_ratio/(len(E)-2)	

E=np.linalg.eigvals(A.dot(circuit.dot(A_inv)))
ph_array=np.sort(np.angle(E))

avg_ratio=0
for j in range(len(E)-2):
    d1=ph_array[j+1]-ph_array[j]
    d2=ph_array[j+2]-ph_array[j+1]
    avg_ratio=avg_ratio+abs(min(d1,d2)/max(d1,d2))
avg_ratio_proj=avg_ratio/(len(E)-2)

E=np.linalg.eigvals(circuit)
ph_array=np.sort(np.angle(E))

avg_ratio=0
for j in range(len(E)-2):
    d1=ph_array[j+1]-ph_array[j]
    d2=ph_array[j+2]-ph_array[j+1]
    avg_ratio=avg_ratio+abs(min(d1,d2)/max(d1,d2))
avg_ratio=avg_ratio/(len(E)-2)

    
path1='/mmfs1/storage/users/harkind/Parity_sym_cue_flo/q' + str(q)+'/tr'+str(0)+'/L'+str(L)
np.savez(os.path.join(path1,'Parity_L'+ str(L)+ '_q'+ str(q)+ '_qt'+ str(qt)+'.npz'),sff,avg_ratio_even,avg_ratio_odd,avg_ratio_proj,avg_ratio)