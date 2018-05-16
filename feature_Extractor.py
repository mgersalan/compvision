import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import gzip


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f,encoding='latin1')

print (train_set[0].shape)
print (train_set[1].shape)
print (train_set[1][0])

print (train_set[0][0].shape)

#np.zeros 256 histogram

multip_matrix = np.matrix([[1, 2, 4], [128, 0,8],[64,32,16]])
print (multip_matrix)

def create_lbp_mat(mat):
    
    R,C = mat.shape
    
    transformed = np.zeros((26, 26))
    for i in range (1,R-1):
        for k in range (1,C-1):
            
            binary_matrix = np.zeros((3, 3))
            sub_mat = mat[i-1:i+2,k-1:k+2]
            mid = sub_mat[1,1]
            for t in range(0,2):
                for j in range(0,2):
                    if sub_mat[t,j] > mid:
                        binary_matrix[t,j] = 1
            transformed[i-1,k-1] = np.sum(np.multiply(multip_matrix,binary_matrix))  

    #print (transformed)
    return transformed



training = np.zeros((1000,256)).astype(np.float32)
test = np.zeros((100,256)).astype(np.float32)

for i in range (0,1000):
    res = create_lbp_mat(train_set[0][i].reshape((28, 28)))
    
    unique, counts = np.unique(res, return_counts=True)
    dictionary = dict(zip(unique, counts))
    
    for k in range(0,255):
        if k in unique:
            
            training[i,k] = dictionary[k]
            
        else:
            training[i,k] = 0

for i in range (0,100):
    res = create_lbp_mat(test_set[0][i].reshape((28, 28)))
    
    unique, counts = np.unique(res, return_counts=True)
    dictionary = dict(zip(unique, counts))
    
    for k in range(0,255):
        if k in unique:
            test[i,k] = dictionary[k] 
        else:
            test[i,k] = 0
    

knn = cv2.ml.KNearest_create()

responses = train_set[1][0:1000]
test_responses = test_set[1][0:100]

print ('res shape',responses.shape)
print ('train shape ',training.shape)
print (training[0])

knn.train(training.astype(np.float32),cv2.ml.ROW_SAMPLE,responses)

#print (test[0])


ret, results, neighbours, dist = knn.findNearest(test.astype(np.float32), 9)

counter = 0
for i in range(0,100):
    if results[i] == test_responses[i]:
        counter = counter + 1

print (counter/100)

#print (np.asarray((unique, counts)).T)

#print (feature_hist)

#print (a)
#hist = np.histogram(a)

