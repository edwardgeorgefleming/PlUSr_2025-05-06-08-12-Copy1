#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from pmlb import fetch_data, classification_dataset_names
import time

def create_X_Y(m,n, p_Noise_1, p_Noise_2):
    
    # Fill X randomly with ones and zeros
    X = np.random.randint(2, size=(m,n))
    
    # Find Y with the following Boolean expression for X
    Y = np.zeros((m,1), dtype=int)
        
    Noise_1 = np.random.randint(int(1 / p_Noise_1), size=m)
    # So, the probability of getting a zero in one samples of all of Noise1 is pNoise1
    
    Noise_2 = np.random.randint(int(1 / p_Noise_2), size=m)
    # So, the probability of getting a zero in one samples of Noise2 is pNoise2
    
    Noise_n = np.random.randint(n, size=m)
    # So, the probability of getting a zero in one samples of Noise2 is pNoise2
    
    for i in range(m):
        Y[i,0] = ( not X[i, 90] and not X[i, 91] and X[i, 92] and not X[i, 93] and not X[i, 94] and not X[i, 95] ) and ( ( X[i, 96] or X[i, 104] ) or ( ( not X[i, 6] and not X[i, 9] and not X[i, 15] and not X[i, 34] and not X[i, 40] and not X[i, 76] and not X[i, 83] and X[i, 84] and not X[i, 85] and not X[i, 86] and not X[i, 87] and not X[i, 88] and X[i, 89] and not X[i, 90] and not X[i, 91] and X[i, 92] and not X[i, 93] and not X[i, 94] and not X[i, 95] and not X[i, 97] and X[i, 98] and X[i, 99] and not X[i, 101] and not X[i, 106] and not X[i, 108] and not X[i, 117] and not X[i, 124] and not X[i, 126] and X[i, 128] and not X[i, 132] and not X[i, 135] and not X[i, 136] and not X[i, 138] and not X[i, 141] and not X[i, 145] and not X[i, 147] and X[i, 149] and not X[i, 160] and not X[i, 166] and not X[i, 173] and not X[i, 177] ) ) )
        if Noise_1[i] == 0:
            Y[i,0] = 1 - Y[i,0]    # Make an error in Y one time every Noise1
        if Noise_2[i] == 0:
            X[i,Noise_n] = 1 - X[i,Noise_n]     # Make an error in X somewhere in the line every Noise2
    
    return X, Y


def reset(i,m,n):
    global TX, TnX, TXY, TXnY, TnXY, TnXnY, op_and, and_n, op_or, or_n
    
    # Reset the ith PLUSr's Totals to zero
    TX[i,:] = np.zeros((1,n), dtype=int)
    TnX[i,:] = np.zeros((1,n), dtype=int)
    TXY[i,:] = np.zeros((1,n), dtype=int)
    TnXY[i,:] = np.zeros((1,n), dtype=int)
    TXnY[i,:] = np.zeros((1,n), dtype=int)
    TnXnY[i,:] = np.zeros((1,n), dtype=int)
    
    # Reset the ith PLUSr's Boolean expression to zero
    op_and[i,:] = np.zeros((1,n), dtype=int)
    and_n[i,:] = np.zeros((1,n), dtype=int)
    op_or[i,:] = np.zeros((1,n), dtype=int)
    Tor_n[i,:] = np.zeros((1,n), dtype=int)
    
    return

def initialize_PLUSr(nm, m, n):
    global L, TX, TnX, TXY, TXnY, TnXY, TnXnY, op_and, and_n, op_or, or_n

    # Initialize the Learning Control input to all ones
    L = np.ones((m,1), dtype=int)
    
    # Initialize all PLUSrs' Totals to zero
    TX = np.zeros((nm,n), dtype=int)
    TnX = np.zeros((nm,n), dtype=int)
    TXY = np.zeros((nm,n), dtype=int)
    TnXY = np.zeros((nm,n), dtype=int)
    TXnY = np.zeros((nm,n), dtype=int)
    TnXnY = np.zeros((nm,n), dtype=int)
    
    # Initialize all PLUSrs' Boolean expressions to zero
    op_and = np.zeros((nm,n), dtype=int)
    and_n = np.zeros((nm,n), dtype=int)
    op_or = np.zeros((nm,n), dtype=int)
    or_n = np.zeros((nm,n), dtype=int)
    
    return


def reset_count(i, nm, m, n, X, LL, Y, beta):
    global TX, TnX, TXY, TXnY, TnXY, TnXnY, op_and, and_n, op_or, or_n
    
    L = np.reshape(LL, (m,1))
    truthTXY = np.zeros((nm, n), dtype=int)
    truthTXnY = np.zeros((nm, n), dtype=int)
    truthTnXY = np.zeros((nm, n), dtype=int)
    truthTnXnY = np.zeros((nm, n), dtype=int)
    
    # Reset and find the Totals of the ith PLUSr
    TX[i,:] = np.reshape(np.sum(X*L, axis=0), (1,n))
    TnX[i,:] = np.reshape(np.sum((1-X)*L, axis=0), (1,n))
    TXY[i,:] = np.reshape(np.sum(X*Y*L, axis=0), (1,n))
    TnXY[i,:] = np.reshape(np.sum((1-X)*Y*L, axis=0), (1,n))
    TXnY[i,:] = np.reshape(np.sum(X*(1-Y)*L, axis=0), (1,n))
    TnXnY[i,:] = np.reshape(np.sum((1-X)*(1-Y)*L, axis=0), (1,n))
    
    for j in range(n):
        truthTXY[i,j] = (TXY[i,j] >= (TX[i,j] - TX[i,j] // beta[i]) > 0)
        truthTXnY[i,j] = (TXnY[i,j] >= (TX[i,j] - TX[i,j] // beta[i]) > 0)
        truthTnXY[i,j] = (TnXY[i,j] >= (TnX[i,j] - TnX[i,j] // beta[i]) > 0)
        truthTnXnY[i,j] = (TnXnY[i,j] >= (TnX[i,j] - TnX[i,j] // beta[i]) > 0)
    
    # Rest and make the Boolean expression for the ith PLUsr
    op_and[i,:] = np.reshape(truthTXnY[i,:] + truthTnXnY[i,:], (1,n))
    and_n[i,:] = np.reshape(truthTnXnY[i,:], (1,n))
    op_or[i,:] =  np.reshape(truthTXY[i,:] + truthTnXY[i,:], (1,n))
    or_n[i,:] = np.reshape(truthTnXY[i,:], (1,n))
    
    return
        

def PLUSr(i, nm, m, n, X):
    global TX, TnX, TXY, TXnY, TnXY, TnXnY, op_and, and_n, op_or, or_n
    
    n_Z = np.ones((m,nm), dtype=int)
    Z = np.zeros((m,nm), dtype=int) 
    
    # Find the output Z for the ith PLUSr's Boolean expression 
    for k in range(m):
        start = True
        
        # First check to see if there are any logical ANDs  (Remember we are going to use De Morgan's Theorem)
        for j in range(n):
            if op_and[i,j] >= 1 and start:
                n_Z[k,i] = (X[k,j] != (and_n[i,j] > 0))
                start = False
            elif op_and[i,j] >= 1:
                n_Z[k,i] = n_Z[k,i] or (X[k, j] != (and_n[i,j] > 0))
                
        if not start:
            Z[k,i] = not n_Z[k,i]
        
        # Second check to see if there are any logical ORs
        for j in range(n):
            if op_or[i,j] >= 1 and start:
                Z[k,i] = X[k,j] != (or_n[i,j] > 0)
                start = False
            elif op_or[i,j] >= 1:
                Z[k,i] = Z[k,i] or X[k, j] != (or_n[i,j] > 0)
              
    return Z[:,i]


def print_PLUSr_Bool_Eq(i, m, n):
    global op_and, and_n, op_or, or_n
    
    # Check to see if there are any Boolean expression for the ith PLUSr
    if np.all(op_and[i,:] == 0) and np.all(op_or[i,:] == 0):
        print('0', end = '')
        return
    
    # Check to see if there are any logical ORs for the ith PLUSr, if so print them
    count = 0
    for j in range(n):
        count = count + 1 if op_or[i,j] >= 1 else count
    count_save = count
    if 0 < count <= 200:
        for j in range(n):
            if op_or[i,j] >= 1:
                count -= 1
                if or_n[i,j]:
                    print('not ', end = '')
                print('X[i,',j, end = '')
                if count <= 0:
                    print(']', end = '')
                else:
                    print('] or ', end = '')
    elif count > 200:
        print(' > 200 Variables', end = '')
    
    # Check to see if there are any logical ANDs for the ith PLUSr, if so print them
    count = 0
    for j in range(n):
        count = count + 1 if op_and[i,j] >= 1 else count
    if count >= 1 and count_save >= 1:
        print(' Or (', end = '')
    if count >= 1:
        if 0 < count <= 200:
            for j in range(n):
                if op_and[i,j] >= 1:
                    count -= 1
                    if 1 - and_n[i,j]:
                        print('not ', end = '')
                    print('X[i,',j, end = '')
                    if count <= 0:
                        if count_save >= 1:
                            print('])', end = '')
                        else:
                            print(']', end = '')
                    else:
                        print('] and ', end = '')
        elif count > 200:
            print(' > 200 Variables', end = '')
    
    return


def PLUSr_print(i, m, Y, Z):
    global y, beta, aamax, beta_save, i_max
    
    # Check to see if the accuracy of all of the i PLUSrs have impoved, if so print all of the i PLUSrs' Boolean Expressions
    winner = 0    
    aa = np.sum(join(i, m, Z) == Y)

    if aa > aamax:
        print()
        print('DNA Category =', y)
        print('beta =', beta)
        i_max = i
        aamax = aa
        beta_save = np.copy(beta)
        print('Test 1 Accuracy =', aa, end = '')
        print('/', m, end = '')
        print(' =', aa/m)

        print("The", i+1
              , end = "")
        print(" PLUSr's ", end = '')
        print('Boolean Expression = ', end = '')
        if i >= 0:
            print('( ', end = '')
            print_PLUSr_Bool_Eq(0, m, n)
        if i >= 1:
            print(' ) and ( ( ', end = '')
            print_PLUSr_Bool_Eq(1, m, n)
        if i >= 2:
            print(' ) or ( ( ', end = '')
            print_PLUSr_Bool_Eq(2, m, n)
        if i >= 3:
            print(' ) and ( ( ', end = '')
            print_PLUSr_Bool_Eq(3, m, n)
        if i >= 4:
            print(' ) or ( ( ', end = '')
            print_PLUSr_Bool_Eq(4, m, n)
        if i >= 5:
            print(' ) and ( ( ', end = '')
            print_PLUSr_Bool_Eq(5, m, n)
        if i >= 6:
            print(' ) or ( ( ', end = '')
            print_PLUSr_Bool_Eq(6, m, n)
        if i >= 7:
            print(' ) and ( ( ', end = '')
            print_PLUSr_Bool_Eq(7, m, n)
        if i >= 8:
            print(' ) or ( ( ', end = '')
            print_PLUSr_Bool_Eq(8, m, n)
        if i >= 8:
            print(' )', end = '')
        if i >= 7:
            print(' )', end = '')
        if i >= 6:
            print(' )', end = '')
        if i >= 5:
            print(' )', end = '')
        if i >= 4:
            print(' )', end = '')
        if i >= 3:
            print(' )', end = '')
        if i >= 2:
            print(' )', end = '')
        if i >= 1:
            print(' )', end = '')
        if i >= 0:
            print(' )')
        winner = 1
    return winner


def join(i, m, Z):
    
    # Join all i+1 PLUSrs outputs
    if i == 8:
        Zf = np.reshape(Z[:,0] * (1 - (1 - Z[:,1]) * (1 - Z[:,2] *(1 - (1 - Z[:,3])*(1-Z[:,4]*(1 - (1 - Z[:,5])*(1 - Z[:,6]*(1 - (1 - Z[:,7])*(1 - Z[:,8])))))))), (m,1))
    if i == 7:
        Zf = np.reshape(Z[:,0] * (1 - (1 - Z[:,1]) * (1 - Z[:,2] *(1 - (1 - Z[:,3])*(1-Z[:,4]*(1 - (1 - Z[:,5])*(1 - Z[:,6]*Z[:,7])))))), (m,1))
    if i == 6:
        Zf = np.reshape(Z[:,0] * (1 - (1 - Z[:,1]) * (1 - Z[:,2] *(1 - (1 - Z[:,3])*(1-Z[:,4]*(1 - (1 - Z[:,5])*(1 - Z[:,6])))))), (m,1))
    elif i == 5:
        Zf = np.reshape(Z[:,0] * (1 - (1 - Z[:,1]) * (1 - Z[:,2] *(1 - (1 - Z[:,3])*(1-Z[:,4]*Z[:,5])))), (m,1))
    elif i == 4:
        Zf = np.reshape(Z[:,0] * (1 - (1 - Z[:,1]) * (1 - Z[:,2] *(1 - (1 - Z[:,3])*(1-Z[:,4])))), (m,1))
    elif i == 3:
        Zf = np.reshape(Z[:,0] * (1 - (1 - Z[:,1]) * (1 - Z[:,2] *Z[:,3])), (m,1))
    elif i == 2:
        Zf = np.reshape(Z[:,0] * (1 - (1 - Z[:,1]) * (1 - Z[:,2])), (m,1))
    elif i == 1:
        Zf = np.reshape(Z[:,0] * Z[:,1], (m,1))
    elif i == 0:
        Zf = np.reshape(Z[:,0], (m,1))
            
    return Zf
    
    
def check(i, m, Z, Y, aamax):
    
    # Check the accuracy for all the i+1 PLUSrs
    aa = np.sum(join(i, m, Z) == Y)
    print('Test 2 Accuracy =', aa, end = '')
    print('/', m, end = '')
    print(' =', aa/m)
    print('Total Time =', (time.time() - start_time), end = '')
    print(' seconds =', (time.time() - start_time)  / 86400, end = '')
    print(' days')     
    
    return


def test_print(i):
    global nm, n, m_test1, m_test2, test1_X, test1_Y, test2_Y, test1_Z, test2_Z
    
    # Get the outputs for all the i+1 PLUSrs for Test 1
    for ii in range(i+1):
        test1_Z[:,ii] = PLUSr(ii, nm, m_test1, n, test1_X)

    # Check and see if the accuracy of Test 1 is better, if so do Test 2 and find it's accuracy then print
    w = PLUSr_print(i, m_test1, test1_Y, test1_Z)
    if w:
        for ii in range(i+1):
            test2_Z[:,ii] = PLUSr(ii, nm, m_test2, n, test2_X)
        check(i, m_test2, test2_Z, test2_Y, aamax)
        
    return


# Read in the datasets and split them into training/testing1/testing2

dna_X, dna_y = fetch_data('dna', return_X_y=True, local_cache_dir='C:/Users/edwar/Documents/PLUSr/dataset')
train_X_org, test1_X, train_y_org, test1_y_org = train_test_split(dna_X, dna_y, random_state=0)
train_X, test2_X, train2_y_org, test2_y_org = train_test_split(train_X_org, train_y_org, random_state=0)
train_y = np.reshape(train2_y_org, train_X.shape[0])
test1_y = np.reshape(test1_y_org, test1_X.shape[0])
test2_y = np.reshape(test2_y_org, test2_X.shape[0])

m = train_X.shape[0]                                 # m = number of lines in X
m_test1 = test1_X.shape[0]                           # m_test1 = number of lines in Test 1
m_test2 = test2_X.shape[0]                           # m_test2 = number of lines in Test 2
n = train_X.shape[1]                                 # n = number of environmental input on the PLUSr
'''
m = 600
m_test1 = 500
m_test2 = 500
n = 180
'''
nm = 10                                              # nm = number of PLUSrs in the string
beta = np.ones(nm, dtype=int)                        # beta = a matrix of betas
beta_save = np.ones(nm, dtype=int)                   # beta_save = save a matrix of betas
base = 2
Z = np.zeros((m,nm), dtype=int)                      # Z + a matrix of the PLUSr's outputs
test1_Z = np.zeros((m_test1,nm), dtype=int)          # test1_Z = a matrix of the PLUSr's outputs for Test 1
test2_Z = np.zeros((m_test2,nm), dtype=int)          # test1_Z = a matrix of the PLUSr's outputs for Test 2
limit = int(input('Enter an integer 1 through 9 to limit the number of PLUSrs in a chaining series configuration = '))
if limit < 1 or limit > 9:
    print('Error')

aamax = 0
print('Start')

start_time = time.time()
for category in range(1,4):
    initialize_PLUSr(nm, m, n)
    y = category
    
    '''
    
    # Uncomment this section to test the Program
    y = 1
    # Add 1% and 0.1% Error in White Noise in the Data
    p_Noise_1 = 0.001   # 0.1% Error in Y
    p_Noise_2 = 0.01     # Data with 1% Error in X 
    train_X, train_y = create_X_Y(m,n, p_Noise_1, p_Noise_2)
    test1_X, test1_y = create_X_Y(m_test1,n, p_Noise_1, p_Noise_2)
    test2_X, test2_y = create_X_Y(m_test2,n, p_Noise_1, p_Noise_2)
    
    '''
    
    train_Y = np.reshape(train_y==y, (m, 1))
    test1_Y = np.reshape(test1_y==y, (m_test1, 1))
    test2_Y = np.reshape(test2_y==y, (m_test2, 1))
    
    i_max = 0
    aamax = 0
    for i in range(limit):
        # Get beta for first PLUSr in the string of PLUSrs  and  use a range of 1 to test the program
        if np.any(L == 1):
            for i_beta in range(10):
                beta[i] = m // (base ** i_beta)
                # Get Z for the first PLUSr with L = 1
                reset_count(i, nm, m, n, train_X, L, train_Y, beta)

                test_print(i)
            beta = np.copy(beta_save)
            reset_count(i, nm, m, n, train_X, L, train_Y, beta)
            Z[:,i] = PLUSr(i, nm, m, n, train_X)
            if i // 2 == i:
                L = Z[:,i]
            else:
                L = 1 - Z[:,i]      
print()
print('Total Time =', (time.time() - start_time), end = '')
print(' seconds =', (time.time() - start_time)  / 86400, end = '')
print(' days')
print()      
print('Done')



# In[ ]:





# In[ ]:




