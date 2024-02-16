############ General top level library, includes the definition of 1/2 body gates and construction of circuit for brick wall model, random phase model, kicked Ising model and their duals


########### For n-by-n CUE 

# cue(n)

########### For brick wall geometry CUE, with local on-site dimension q and size L, use [For pbc, choose pbc_boo = yes] 

# W_cue_brickwall(q,L,pbc_boo=True)   

########### For Random Phase model, with local on-site dimension q and size L, 
########### and with standard deviation epsilon for the random phase gate

# W_cue_random_phase(q,L,eps,pbc_boo=True)


import scipy as sp
import scipy.linalg as linalg
import timeit
#import matplotlib
#import matplotlib.pyplot
#import matplotlib.pyplot as plt
import scipy.sparse.linalg
import numpy as np
import numpy.linalg
import pickle
import re
import scipy.sparse as sparse
import math
from optparse import OptionParser
import itertools
import scipy.sparse.linalg as splinalg
from random import randint
from scipy.special import binom
import random
import math


def cue(n):
    """A random matrix distribute with Haar measure of size n x n"""
    z = (sp.randn(n,n) + 1j * sp.randn(n,n))/sp.sqrt(2.)
    q,r = linalg.qr(z)
    d = np.diagonal(r)
    ph = d/sp.absolute(d)
    return np.multiply(q,ph)


def cue_u1(n): #only works for q=2, n is redundant, kept for convenience in using funn(funnarg) format later on
    """U(1) conserved unitary with q=2, with each block a cue"""
    tmp1=cue(1)
    tmp2=cue(1)
    tmp3=cue(2)
    #return sparse.block_diag((tmp1,tmp3,tmp2))
    return linalg.block_diag(tmp1,tmp3,tmp2)


# random gate with Z2 conservation
def cue_z2(n): #only works for q=2, n is redundant, kept for convenience in using funn(funnarg) format later on
    """Z2 conserved unitary with q=2, with each block a cue"""
    tmp1=cue(2)
    tmp2=cue(2)
    #return sparse.block_diag((tmp1,tmp3,tmp2))
    mat1=linalg.block_diag(tmp1[0,0],tmp2,tmp1[1,1])
    mat1[0,3]=tmp1[0,1]
    mat1[3,0]=tmp1[1,0]
    return mat1



############################## For Periodic boundary condition, need to shift the indices of the last matrix

def shiftinteger(n, q, t):
    """shift on the left the digits of an integer n with t components in base q"""
    # Given a vector (a_1, a_2, a_3,... a_t), where a_i=1...t
    # We represent these vector as a single number by writing
    # n = sum_{i=1}^{t} a_i q^{i-1}
    # Example: q=10,  then  (1,2,3) ==> 321
    n1 = n*q
    #print(n1)
    at = n1 // q**t  #// means floor division
    #print(at)
    return n1 - at * (q**t) + at

def shiftmatrix(q,t):
    """matrix performing the shift on the left in the space of dimension q**t: a1 a2... at -> a2 a3... at a1"""
    #in the sparse2gate code we will transpose the matrix to use with the convention of q-bit numbers arranged as at at-1 ... a2 a1
    data = np.ones(q**t)
    #print(data)
    ii = range(q**t) #ROW LABELs
    #print(ii)
    jj = [shiftinteger(n, q, t) for n in ii]  
    #COLUMN LABELs. These exchange a1 a2... at -> a2 a3... at a1
    #jj= [shiftinteger(0, 2, 2), shiftinteger(1, 2, 2), shiftinteger(2, 2, 2),...]
    #print(jj)
    return sparse.coo_matrix((data, (ii, jj))) 



############################## General gate of form  "1 \otimes (2gate) \otimes 1 \otimes 1" (Embed 1/2 site gates into the full Hilbert space)

def Sparse1gate_general_multiarg(onegate_func,q,L,i,*args):
    onegate = onegate_func(*args)
    """i is the position of the left site of the 2-gate"""
    if (L==1):
        return sp.sparse.csr_matrix(onegate)
    else:
        temp= sp.sparse.kron(sp.sparse.identity(q**(i-1)),sp.sparse.csr_matrix(onegate))
        temp = sp.sparse.kron(temp,sp.sparse.identity(q**(L-i))) 
    return temp

def Sparse2gate_general_multiarg(twogate_func,q,L,i,*args):
    twogate = twogate_func(*args)
    """i is the position of the left site of the 2-gate"""
    #print(twogate.todense())
    if (L==2):
        return sp.sparse.csr_matrix(twogate)
    else:
        if (i==L):
            temp= sp.sparse.kron(sp.sparse.identity(q**(L-2)),sp.sparse.csr_matrix(twogate))
            SM = shiftmatrix(q, L)
            SMT= SM.transpose()
            temp = temp.dot(SMT)
            temp = SM.dot(temp)
        else:
            temp= sp.sparse.kron(sp.sparse.identity(q**(i-1)),sp.sparse.csr_matrix(twogate))
            temp = sp.sparse.kron(temp,sp.sparse.identity(q**(L-i-1))) 
    return temp


############################### Embed the gates into full Hilbert space when we have translational invariance, so use the same matrix


def Sparse1gate_general_multiarg_given_mat(onegate,q,L,i,): #onegate is now the matrix instead of function to generate the matrix
    #onegate = onegate_func(*args)
    """i is the position of the left site of the 2-gate"""
    if (L==1):
        return sp.sparse.csr_matrix(onegate)
    else:
        temp= sp.sparse.kron(sp.sparse.identity(q**(i-1)),sp.sparse.csr_matrix(onegate))
        temp = sp.sparse.kron(temp,sp.sparse.identity(q**(L-i))) 
    return temp

def Sparse2gate_general_multiarg_given_mat(twogate,q,L,i):
    #twogate = twogate_func(*args)
    """i is the position of the left site of the 2-gate"""
    #print(twogate.todense())
    if (L==2):
        return sp.sparse.csr_matrix(twogate)
    else:
        if (i==L):
            temp= sp.sparse.kron(sp.sparse.identity(q**(L-2)),sp.sparse.csr_matrix(twogate))
            SM = shiftmatrix(q, L)
            SMT= SM.transpose()
            temp = temp.dot(SMT)
            temp = SM.dot(temp)
        else:
            temp= sp.sparse.kron(sp.sparse.identity(q**(i-1)),sp.sparse.csr_matrix(twogate))
            temp = sp.sparse.kron(temp,sp.sparse.identity(q**(L-i-1))) 
    return temp




########################################################################## Quantum Circuit, with Brick wall geometry###############################################
##########################################################################          BRICK WALL MODEL                ###############################################
####### 1st layer, 2gates on odd sites, CUE
####### 2nd layer, 2gates on even sites, CUE

def W_cue_brickwall(q,L,list=True,pbc_boo=True):
    twogate_func= cue
    func_arg=q**2
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L+1,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L+1,2)]
    else:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L,2)]

    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
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

########################### charge conserved Brick wall circuit in time
###########################


def W_cue_brickwall_u1(q,L,list=True,pbc_boo=True):
    twogate_func= cue_u1
    func_arg=q**2
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L+1,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L+1,2)]
    else:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L,2)]

    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
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



########################### cue Brick wall circuit with TRANSLATIONAL INVARIANCE
###########################


############ AA configuration has consecutive timesteps with the same unitary
############ AB has alternating in time step unitaries
def W_cue_brickwall_AA_trans_inv(q,L,list=True,pbc_boo=True):
    twogate_A= cue(q**2)
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate_A,q,L,i) for i in range(1,L+1,2)]
        biglist2=[Sparse2gate_general_multiarg_given_mat(twogate_A,q,L,i) for i in range(2,L+1,2)]
    else:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate_A,q,L,i) for i in range(1,L,2)]
        biglist2=[Sparse2gate_general_multiarg_given_mat(twogate_A,q,L,i) for i in range(2,L,2)]

    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
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

def W_cue_brickwall_AB_trans_inv(q,L,list=True,pbc_boo=True):
    twogate_A= cue(q**2)
    twogate_B= cue(q**2)
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate_A,q,L,i) for i in range(1,L+1,2)]
        biglist2=[Sparse2gate_general_multiarg_given_mat(twogate_B,q,L,i) for i in range(2,L+1,2)]
    else:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate_A,q,L,i) for i in range(1,L,2)]
        biglist2=[Sparse2gate_general_multiarg_given_mat(twogate_B,q,L,i) for i in range(2,L,2)]

    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
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





########################### Z2 parity conserved Brick wall circuit in time
###########################


def W_cue_brickwall_z2(q,L,list=True,pbc_boo=True):
    twogate_func= cue_z2
    func_arg=q**2
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L+1,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L+1,2)]
    else:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L,2)]

    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
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







########################### DUAL OF BRICK WALL CIRCUIT
########################### Constructing dual of temporal random spatially invariant lists, both AA and AB type (both type and TITR meant in ordinary circuit)

############ First need to define dual gates

def index_swapped_dual_unitary(q): #create a random unitary with swapped indices of the sparse unitary matrix  
    m1=cue(q**2)
    #print("trace original",np.trace(m1),np.abs(np.linalg.det(m1)))
    m2=np.reshape(m1,(q,q,q,q),order='C') #first reshape into 
    #print(m2.shape)
    m2=np.transpose(m2,(2,3,0,1)) #first swap two indices

    m2=np.transpose(m2,(0,2,1,3)) #second swap, see notes

    m2=np.reshape(m2,(q**2,q**2),order='C') #checked that removing transpose and reshaping again using this line leads to the same matrix
    return m2

def index_swapped_dual_u1_unitary(q): #create a charge conserved unitary and swap indices to the dual matrix  
    m1=cue_u1(q**2)
    #print("trace original",np.trace(m1),np.abs(np.linalg.det(m1)))
    m2=np.reshape(m1,(q,q,q,q),order='C') #first reshape into 
    #print(m2.shape)
    m2=np.transpose(m2,(2,3,0,1)) #first swap two indices

    m2=np.transpose(m2,(0,2,1,3)) #second swap, see notes

    m2=np.reshape(m2,(q**2,q**2),order='C') #checked that removing transpose and reshaping again using this line leads to the same matrix
    return m2

def index_swapped_dual_z2_unitary(q): #create a charge conserved unitary and swap indices to the dual matrix  
    m1=cue_z2(q**2)
    #print("trace original",np.trace(m1),np.abs(np.linalg.det(m1)))
    m2=np.reshape(m1,(q,q,q,q),order='C') #first reshape into 
    #print(m2.shape)
    m2=np.transpose(m2,(2,3,0,1)) #first swap two indices

    m2=np.transpose(m2,(0,2,1,3)) #second swap, see notes

    m2=np.reshape(m2,(q**2,q**2),order='C') #checked that removing transpose and reshaping again using this line leads to the same matrix
    return m2



def index_swap_given_mat(mat,q): #swap indices of a given q**2xq**2 matrix
    m1=mat
    #print("trace original",np.trace(m1),np.abs(np.linalg.det(m1)))
    m2=np.reshape(m1,(q,q,q,q),order='C') #first reshape into 
    #print(m2.shape)
    m2=np.transpose(m2,(2,3,0,1)) #first swap two indices

    m2=np.transpose(m2,(0,2,1,3)) #second swap, see notes

    m2=np.reshape(m2,(q**2,q**2),order='C') #checked that removing transpose and reshaping again using this line leads to the same matrix
    return m2


############ Building dual cricuit when temporally random AB type


def W_dual_brickwall_AB_trans_inv(q,L,list=True,pbc_boo=True): #using part of the normal W_cue code without translation invariance since dual circuit doesn't have one
    twogate_func= index_swapped_dual_unitary #instead of cue, swapped unitary
    func_arg=q #q**2 argument for cue, but for swapped unitary, argument q
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L+1,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L+1,2)]
    else:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L,2)]
    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
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



def W_dual_brickwall_AB_trans_inv_multi_t(q,L,tMax,pbc_boo=True): #return W^t (do we really need this function)
    twogate_func= index_swapped_dual_unitary #instead of cue, swapped unitary
    func_arg=q #q**2 argument for cue, but for swapped unitary, argument q
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L+1,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L+1,2)]
    else:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L,2)]
    temp = biglist1[0] #picking the first matrix to start multiplying

    for (i,mat) in enumerate(biglist1):
        if not(i==0): #since first matrix already chosen
            temp = mat.dot(temp)
        else:
            pass
    for (j,mat) in enumerate(biglist2):
        temp = mat.dot(temp)



    for tt in range(tMax-1):
        for (i,mat) in enumerate(biglist1): #removing the not i =0 because the new rows are multiplying the existing ones
                temp = mat.dot(temp)
        for (j,mat) in enumerate(biglist2):
            temp = mat.dot(temp)

    return temp.todense()


########### Dual circuit using the charge conserved unitaries cue_u1

def W_dual_brickwall_u1_AB_trans_inv(q,L,list=True,pbc_boo=True): #using part of the normal W_cue code without translation invariance since dual circuit doesn't have one
    twogate_func= index_swapped_dual_u1_unitary #instead of cue, swapped u1 unitary
    func_arg=q #q**2 argument for cue, but for swapped unitary, argument q
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L+1,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L+1,2)]
    else:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L,2)]
    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
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

########### Dual circuit using the z2 conserved unitaries cue_z2

def W_dual_brickwall_z2_AB_trans_inv(q,L,list=True,pbc_boo=True): #using part of the normal W_cue code without translation invariance since dual circuit doesn't have one
    twogate_func= index_swapped_dual_z2_unitary #instead of cue, swapped u1 unitary
    func_arg=q #q**2 argument for cue, but for swapped unitary, argument q
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L+1,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L+1,2)]
    else:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(1,L,2)]
        biglist2=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg) for i in range(2,L,2)]
    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
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


############ Building dual cricuit when temporally floquet AB type

def W_dual_Floquet_TI_brickwall_AB_trans_inv(q,L,list=True,pbc_boo=True): #above function but with returning as dense matrix, for purpose of single row computation

    twogate_func= index_swapped_dual_unitary #instead of cue, swapped unitary

    #func_arg=q #q**2 argument for cue, but for swapped unitary, argument q
    twogate_A= twogate_func(q)
    twogate_B= twogate_func(q)

    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate_A,q,L,i) for i in range(1,L+1,2)]
        biglist2=[Sparse2gate_general_multiarg_given_mat(twogate_B,q,L,i) for i in range(2,L+1,2)]
    else:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate_A,q,L,i) for i in range(1,L,2)]
        biglist2=[Sparse2gate_general_multiarg_given_mat(twogate_B,q,L,i) for i in range(2,L,2)]

    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
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



##########################################################################           Quantum Circuit              ###############################################
##########################################################################          RANDOM PHASE MODEL            ###############################################


############################## Quantum Circuit, random phase model
####### 1st layer, 1gates, CUE (use definition from brick wall model)
####### 2nd layer, 2gates Diagonal random phases

def TwoGateRanPhase(q,eps): #defining the two site gate with random phases on the diagonal
    """eps = Standard deviation """
    ranphases = np.random.normal(0, eps, q*q)
    #ranphases = 2*np.pi*np.random.rand(q*q)
    diagonals = np.array([[numpy.exp(1j *ranphases[i]) for i in range(0,q*q)]])
    return scipy.sparse.diags(diagonals, [0])


###### Construct the layers

def W_cue_random_phase(q,L,eps,list=True, pbc_boo=True):
    twogate_func= TwoGateRanPhase
    func_arg1=q
    func_arg2= eps
    
    ### Random Phase
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg1,func_arg2) for i in range(1,L+1)]
    else:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg1,func_arg2) for i in range(1,L)]
        
    ### CUE 1gates:
    onegate_func = cue
    func_arg3= q
    biglist2=[Sparse1gate_general_multiarg(onegate_func,q,L,i,func_arg3) for i in range(1,L+1)]
    
    ### Together

    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
        return biglist1+biglist2

    else:

        temp = biglist2[0]
        for (i,mat) in enumerate(biglist2):
            if not(i==0):
                temp = mat.dot(temp)
            else:
                pass
        for (j,mat) in enumerate(biglist1):
            temp = mat.dot(temp)
        return temp.todense()




########################### Random phase model with  TRANSLATIONAL INVARIANCE
###########################




def W_cue_random_phase_trans_inv(q,L,eps,list=True,pbc_boo=True):
    twogate= TwoGateRanPhase(q,eps)
    
    ### Random Phase
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate,q,L,i) for i in range(1,L+1)]
    else:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate,q,L,i) for i in range(1,L)]
        
    ### CUE 1gates:
    onegate = cue(q)
    biglist2=[Sparse1gate_general_multiarg_given_mat(onegate,q,L,i) for i in range(1,L+1)]
    
    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
        return biglist1+biglist2

    else:

        temp = biglist2[0]
        for (i,mat) in enumerate(biglist2):
            if not(i==0):
                temp = mat.dot(temp)
            else:
                pass
        for (j,mat) in enumerate(biglist1):
            temp = mat.dot(temp)
        return temp.todense()





########################### DUAL OF RANDOM PHASE MODEL
#########################

#####Defining general dual of 1 and diagonal 2 gates, does't come directly in use

def index_swap_1site_to_2site_given_mat(mat,q): #conver the single site unitary to the two site diagonal matrix in the dual direction
    m1=mat
    #print("trace original",np.trace(m1),np.abs(np.linalg.det(m1)))
    m2=np.zeros((q,q,q,q),dtype=complex) #define the two site matrix

    for i in range(q):
        for j in range(q):
            m2[i,j,i,j]=m1[j,i] #The diagonal elements are only non zero (U_{a1a2} -> U_{a2a1b2b1})

    m2=np.reshape(m2,(q**2,q**2),order='C') #reshaping to a two site matrix
    return m2

def index_swap_2site_to_1site_given_mat(mat,q): #conver the two site unitary (diagal in RPM/KIM) to the one site  matrix in the dual direction
    m1=mat
    #print("trace original",np.trace(m1),np.abs(np.linalg.det(m1)))
    m1=np.reshape(m1,(q,q,q,q),order='C') #first reshape into 

    m2=np.zeros((q,q),dtype=complex) #define the two site matrix

    for i in range(q):
        for j in range(q):
            m2[i,j]=m1[i,j,i,j] #The diagonal elements are only non zero (U_{a1a2} -> U_{a2a1b2b1})

    return m2


##### Defining duals of 1 and 2 site gates for the random phase model


def rpm_dual_1site_to_2site(q): #conver the single site unitary to the two site diagonal matrix in the dual direction
    m1=cue(q)
    #print("trace original",np.trace(m1),np.abs(np.linalg.det(m1)))
    m2=np.zeros((q,q,q,q),dtype=complex) #define the two site matrix

    for i in range(q):
        for j in range(q):
            m2[i,j,i,j]=m1[j,i] #The diagonal elements are only non zero (U_{a1a2} -> U_{a2a1b2b1})

    m2=np.reshape(m2,(q**2,q**2),order='C') #reshaping to a two site matrix
    return m2

def rpm_dual_2site_to_1site(q,eps): #conver the two site unitary (diagal in RPM/KIM) to the one site  matrix in the dual direction
    m1=TwoGateRanPhase(q,eps)
    #print("trace original",np.trace(m1),np.abs(np.linalg.det(m1)))
    m1=np.reshape(m1.toarray(),(q,q,q,q),order='C') #first reshape into 

    m2=np.zeros((q,q),dtype=complex) #define the two site matrix

    for i in range(q):
        for j in range(q):
            m2[i,j]=m1[i,j,i,j] #The diagonal elements are only non zero (U_{a1a2} -> U_{a2a1b2b1})

    return m2


###### Construct the layers for dual circuit with temporally random gates


def W_dual_random_phase_trans_inv(q,t,eps,list=True,pbc_boo=True): #dual of the translational invariant, temporally random, random phase model
    #twogate= TwoGateRanPhase(q,eps)
    twogate_func= rpm_dual_1site_to_2site
    func_arg1=q
    #func_arg2= eps
    
    ### Random Phase
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,t,i,func_arg1) for i in range(1,t+1)]
    else:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,t,i,func_arg1) for i in range(1,t)]
        
    ### CUE 1gates:
    onegate_func = rpm_dual_2site_to_1site
    func_arg3= q
    func_arg4= eps
    biglist2=[Sparse1gate_general_multiarg(onegate_func,q,t,i,func_arg3,func_arg4) for i in range(1,t+1)]
    
    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
        return biglist1+biglist2

    else:

        temp = biglist2[0]
        for (i,mat) in enumerate(biglist2):
            if not(i==0):
                temp = mat.dot(temp)
            else:
                pass
        for (j,mat) in enumerate(biglist1):
            temp = mat.dot(temp)
        return temp.todense()

###### Construct the layers for dual circuit with temporally Floquet gates

def W_dual_Floquet_random_phase_trans_inv(q,t,eps,list=True,pbc_boo=True): #dual of the translational invariant, temporally random, random phase model
    twogate= rpm_dual_1site_to_2site(q)

    ### Random Phase
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate,q,t,i) for i in range(1,t+1)]
    else:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate,q,t,i) for i in range(1,t)]
        
    ### CUE 1gates:
    onegate = rpm_dual_2site_to_1site(q,eps)

    biglist2=[Sparse1gate_general_multiarg_given_mat(onegate,q,t,i) for i in range(1,t+1)]
    
    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
        return biglist1+biglist2

    else:

        temp = biglist2[0]
        for (i,mat) in enumerate(biglist2):
            if not(i==0):
                temp = mat.dot(temp)
            else:
                pass
        for (j,mat) in enumerate(biglist1):
            temp = mat.dot(temp)
        return temp.todense()



##########################################################################           Quantum Circuit              ###############################################
##########################################################################          KICKED ISING MODEL            ###############################################


############################## First define the corresponding single site and two site gates

# B = transverse field
# h = z field for single site

def singlesite_pros(B, h):
    """W = W2.W1 and 
    W1 = Exp[I h sigma^z_j] . Exp[I B sigma_j^x]
    W2 = Exp[I J sigma^z_j sigma^z_j+1]
    """
    cb = np.cos(B)
    sb = np.sin(B)
    eih = np.exp(- 1j * h)
    #matr = [[eih * cb, -1j *sb/eih ], 
    #        [ -1j*sb* eih, cb/eih]] #for Exp[-I b sigma_x]Exp[-I h sigma_z]
    matr = [[eih * cb, -1j *sb*eih ],
            [ -1j*sb/eih, cb/eih]] #for Exp[-I h sigma_z]Exp[-I b sigma_x]
    return np.array(matr)

### If a single uniform value given for field, convert it to L length list

def makelist(num, L):
    try:
        if (len(num) != L):
            return (num[0:1]*L)
        return num
    except TypeError:
        return [num] * L


###### Two site gate

def TwoGateKickedIsing(J):
    """J = assuming uniform coupling constant """
    return scipy.sparse.diags([np.exp(-1j*J), np.exp(+1j*J),np.exp(+1j*J),np.exp(-1j*J)])


############################## Construct the layers


def W_KickedIsing(q,L,J,B,hm,sigma,list=True,pbc_boo=True):

    twogate= TwoGateKickedIsing(J) #don't need to call two gate everytime, change it
    
    ### Random Phase
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate,q,L,i) for i in range(1,L+1)]
    else:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate,q,L,i) for i in range(1,L)]


    #twogate_func= TwoGateKickedIsing #don't need to call two gate everytime, change it #OLD ONE CALLED TWO GATE EVERY TIME, NOT NEEDED for case B
    #func_arg1=J
    #func_arg2= eps
    
    ### Random Phase
    #if pbc_boo==True:
    #    biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg1) for i in range(1,L+1)]
    #else:
    #    biglist1=[Sparse2gate_general_multiarg(twogate_func,q,L,i,func_arg1) for i in range(1,L)]
        
    ### CUE 1gates:

    onegate_func = singlesite_pros
    Bs = makelist(B, L)
    hs = numpy.random.normal(loc=hm, scale=sigma, size=(L)) 

    func_arg3= Bs
    func_arg4= hs

    biglist2=[Sparse1gate_general_multiarg(onegate_func,q,L,i,func_arg3[i-1],func_arg4[i-1]) for i in range(1,L+1)]
    
    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
        return biglist1+biglist2

    else:

        temp = biglist2[0]
        for (i,mat) in enumerate(biglist2):
            if not(i==0):
                temp = mat.dot(temp)
            else:
                pass
        for (j,mat) in enumerate(biglist1):
            temp = mat.dot(temp)
        return temp.todense()


################################  TRANSLATIONAL INVARIANT version of KIM (disordered field still, but same at each site)

def W_KickedIsing_trans_inv(q,L,J,B,hm,sigma,list=True,pbc_boo=True):


    twogate= TwoGateKickedIsing(J) #don't need to call two gate everytime, change it
    
    ### Random Phase
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate,q,L,i) for i in range(1,L+1)]
    else:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate,q,L,i) for i in range(1,L)]
        
    ### CUE 1gates:

    if(abs(sigma-10*np.pi)<1e-7):
        h = 2*np.pi*numpy.random.rand()
    else: 
        h = numpy.random.normal(loc=hm, scale=sigma) 
    onegate = singlesite_pros(B,h)

    biglist2=[Sparse1gate_general_multiarg_given_mat(onegate,q,L,i) for i in range(1,L+1)]
    
    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
        return biglist1+biglist2

    else:

        temp = biglist2[0]
        for (i,mat) in enumerate(biglist2):
            if not(i==0):
                temp = mat.dot(temp)
            else:
                pass
        for (j,mat) in enumerate(biglist1):
            temp = mat.dot(temp)
        return temp.todense()


################################ DUAL of Kicked Ising model


####first define the dual transformation of one and two site matrices

def KIM_dual_1site_to_2site(B,hm,sigma,q=2): #convert the single site unitary to the two site diagonal matrix in the dual direction

    if(abs(sigma-10*np.pi)<1e-7):
        h = 2*np.pi*numpy.random.rand()
    else: 
        h = numpy.random.normal(loc=hm, scale=sigma) 
    m1=singlesite_pros(B,h)

    #print("trace original",np.trace(m1),np.abs(np.linalg.det(m1)))
    m2=np.zeros((q,q,q,q),dtype=complex) #define the two site matrix

    for i in range(q):
        for j in range(q):
            m2[i,j,i,j]=m1[j,i] #The diagonal elements are only non zero (U_{a1a2} -> U_{a2a1b2b1})

    m2=np.reshape(m2,(q**2,q**2),order='C') #reshaping to a two site matrix
    return m2

def KIM_dual_2site_to_1site(J,q=2): #conver the two site unitary (diagal in RPM/KIM) to the one site  matrix in the dual direction
    m1=TwoGateKickedIsing(J)
    #print("trace original",np.trace(m1),np.abs(np.linalg.det(m1)))
    m1=np.reshape(m1.toarray(),(q,q,q,q),order='C') #first reshape into 

    m2=np.zeros((q,q),dtype=complex) #define the two site matrix

    for i in range(q):
        for j in range(q):
            m2[i,j]=m1[i,j,i,j] #The diagonal elements are only non zero (U_{a1a2} -> U_{a2a1b2b1})

    return m2

#############Using the above to define construct the layers


def W_dual_KickedIsing_trans_inv(q,t,J,B,hm,sigma,list=True,pbc_boo=True): #dual of the translational invariant, temporally random, kicked Ising
    #twogate= TwoGateRanPhase(q,eps)
    twogate_func= KIM_dual_1site_to_2site
    func_arg1=B
    func_arg2= hm
    func_arg3= sigma
    
    ### Random Phase
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,t,i,func_arg1,func_arg2,func_arg3) for i in range(1,t+1)]
    else:
        biglist1=[Sparse2gate_general_multiarg(twogate_func,q,t,i,func_arg1,func_arg2,func_arg3) for i in range(1,t)]
        
    ### CUE 1gates:
    onegate_func = KIM_dual_2site_to_1site
    func_arg3= J
    #func_arg4= eps
    biglist2=[Sparse1gate_general_multiarg(onegate_func,q,t,i,func_arg3) for i in range(1,t+1)]
    
    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
        return biglist1+biglist2

    else:

        temp = biglist2[0]
        for (i,mat) in enumerate(biglist2):
            if not(i==0):
                temp = mat.dot(temp)
            else:
                pass
        for (j,mat) in enumerate(biglist1):
            temp = mat.dot(temp)
        return temp.todense()


###### Construct the layers for dual circuit with temporally Floquet gates

def W_dual_Floquet_KickedIsing_trans_inv(q,t,J,B,hm,sigma,list=True,pbc_boo=True): #dual of the translational invariant, temporally random, random phase model
    #twogate= TwoGateRanPhase(q,eps)
    twogate_func= KIM_dual_1site_to_2site
    func_arg1=B
    func_arg2= hm
    func_arg3= sigma

    twogate=twogate_func(func_arg1,func_arg2,func_arg3)

    ### Random Phase
    if pbc_boo==True:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate,q,t,i) for i in range(1,t+1)]
    else:
        biglist1=[Sparse2gate_general_multiarg_given_mat(twogate,q,t,i) for i in range(1,t)]
        
    ### CUE 1gates:
    onegate_func = KIM_dual_2site_to_1site
    func_arg3= J
    onegate = KIM_dual_2site_to_1site(func_arg3)
    biglist2=[Sparse1gate_general_multiarg_given_mat(onegate,q,t,i) for i in range(1,t+1)]
    
    if(list): #if list is True, returns just the list, otherwise the corresponding multiplied matrix
        return biglist1+biglist2

    else:

        temp = biglist2[0]
        for (i,mat) in enumerate(biglist2):
            if not(i==0):
                temp = mat.dot(temp)
            else:
                pass
        for (j,mat) in enumerate(biglist1):
            temp = mat.dot(temp)
        return temp.todense()


