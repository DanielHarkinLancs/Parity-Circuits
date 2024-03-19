import numpy as np
import scipy as sp
import sys
import os

sys.path.append(r'/mmfs1/home/users/harkind/Random Circuits')

from Lib_gen import *
from Lib_ed import *

os.environ['OMP_NUM_THREADS'] = '1'

L =int(sys.argv[2])        ## Length of the tight binding chain
q=int(sys.argv[1])
tr=int(sys.argv[4])
qt=int(sys.argv[3])
eps=float(sys.argv[5])


def parity_gate(q,eps):
    Q = qSwap(q)
    
    A = np.random.normal(0,eps/4,(q**2,q**2))
    B = np.random.normal(0,eps/4,(q**2,q**2))
    
    h = A+1j*B    
    h = h + (h.conj().T)
    
    gamma = h+(Q.dot(h)).dot(Q)
    
    return sp.linalg.expm(1j*gamma)

def W_coarse_parity(q,eps,L,list=True):
    twogate_func = parity_gate
    func_arg1 = q
    func_arg2 = eps
    onegate_func = cue
    func_arg3 = q
    
    
    ### Random Phase
    if L==2:
        s = sparse.csr_matrix(qSwap(q))
        biglist1 = [Sparse2gate_general_multiarg(twogate_func,q,L,1,func_arg1,func_arg2)]
        
        biglist2 = [Sparse1gate_general_multiarg(onegate_func,q,L,1,func_arg3)]
        biglist2_R = [s.dot(a).dot(s) for a in biglist2]
        biglist2 = biglist2+biglist2_R
    else:
        s = big_swap_matrix(q,L).tocsr()
        
        biglist1 = [Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg1,func_arg2) for i in range(1,int(L/2))]
        biglist1_R = [s.multiply(a).multiply(s) for a in biglist1]
        biglist1 = biglist1+biglist1_R
        
        biglist2 = [Sparse1gate_general_multiarg(onegate_func,q,L,i,func_arg3) for i in range(1,int(L/2))]
        biglist2_R = [s.multiply(a).multiply(s) for a in biglist2]
        biglist2 = biglist2+biglist2_R
    
    ### Together
    if list:
        return biglist1+biglist2
    
    else:
        temp = biglist1[0]
        for (i,mat) in enumerate(biglist1):
            if not(i==0):
                temp = mat.dot(temp)
            else:
                pass
        for (j,mat) in enumerate(biglist2):
            temp = mat.dot(temp)
        return temp.todense()
A = W_coarse_parity(q,eps,L,False)
sff = SFF_Double_Stochastic_Lin_Op(q, L,100,W_coarse_parity,True,q,eps,L,True)

path1='/mmfs1/storage/users/harkind/coarse_temp/q' +str(q)+'/tr' +str(tr)
np.savez(os.path.join(path1,'coarse2'+ '_q'+ str(q)+'_eps'+str(eps)+ '_qt'+ str(qt)+'.npz'),sff)