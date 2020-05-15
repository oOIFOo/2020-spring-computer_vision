from scipy import linalg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.linalg import inv
import argparse

def switch(_input):
    output = _input
    for i in range(_input.shape[0]):
        k = output[i][0]
        output[i][0] = output[i][1]
        output[i][1] = k
    return output
    
def project_matrix(idx):
    if idx == 1:
        object1_point = switch(np.load('object1.npy'))
        object2_point = switch(np.load('object2.npy'))
    if idx == 2:
        object1_point = switch(np.load('object3.npy'))
        object2_point = switch(np.load('object4.npy'))
    tmp = np.ones( (8,8) )
    H = np.ones( (3,3) )
    a = np.zeros( (8,1) )
    
    for i in range(8):
        j = int(i/2)
        if i % 2 == 0:
            for x in range(2):
                tmp[i][x] = object1_point[j][x]
                
            for x in range(3):
                tmp[i][x+3] = 0
                
            for x in range(2):
                tmp[i][x+6] = -1*object2_point[j][0]*object1_point[j][x]
            
        elif i % 2 == 1:
            for x in range(3):
                tmp[i][x] = 0
                
            for x in range(2):
                tmp[i][x+3] = object1_point[j][x]   
            
            for x in range(2):
                tmp[i][x+6] = -1*object2_point[j][1]*object1_point[j][x]
    
    for i in range(8):
        a[i][0] = object2_point[int(i/2)][i%2]
        
    print(tmp)
    Z = np.dot(inv(tmp),a)
    print(Z)
    for x in range(8):
       H[int(x/3)][x%3] =  Z[x][0]
       
    print('H'+str(idx)+' = ')
    print(H)
    return H

def backward1(H):
    object1_point = np.load('object1.npy')
    object2_point = np.load('object2.npy')
    img = cv2.imread('images/1.jpg')
    img_tmp = np.zeros(img.shape, dtype=np.uint8)
    src = np.ones( 3 )
    dst = np.ones( 3 )
    
    mask1 = np.ones(img.shape)                              
    myROI = object1_point  # (x, y)
    cv2.fillPoly(mask1, [np.array(myROI)], 0)
    
    mask2 = np.ones(img.shape)                              
    myROI = object2_point
    cv2.fillPoly(mask2, [np.array(myROI)], 0)

    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i][j][0] == 0:
                dst[0] = i
                dst[1] = j
                src = np.dot(inv(H),dst)
                src = src/src[2]
                #print(dst)
                x = int(round(src[0])) % img.shape[0]
                y = int(round(src[1])) % img.shape[1]
                #print(x,y)
                img_tmp[i][j] = img[x][y]
                
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i][j][0] == 0:
                dst[0] = i
                dst[1] = j
                src = np.dot(H,dst)
                src = src/src[2]
                #print(dst)
                x = int(round(src[0])) % img.shape[0]
                y = int(round(src[1])) % img.shape[1]
                #print(x,y)
                img_tmp[i][j] = img[x][y]
                
    for i in range(img_tmp.shape[0]):
        for j in range(img_tmp.shape[1]):
            if img_tmp[i][j][0] == 0:
                img_tmp[i][j] = img[i][j]
                
    cv2.imwrite('output/1_backward.jpg', img_tmp)

def backward2(H):
    object1_point = np.load('object3.npy')
    object2_point = np.load('object4.npy')
    img1 = cv2.imread('images/2.jpg')
    img2 = cv2.imread('images/3.jpg')
    img_tmp1 = np.zeros(img1.shape, dtype=np.uint8)
    img_tmp2 = np.zeros(img2.shape, dtype=np.uint8)
    src = np.ones( 3 )
    dst = np.ones( 3 )
    
    mask1 = np.ones(img1.shape)                              
    myROI = object1_point
    cv2.fillPoly(mask1, [np.array(myROI)], 0)
    
    mask2 = np.ones(img2.shape)                              
    myROI = object2_point
    cv2.fillPoly(mask2, [np.array(myROI)], 0)
    
    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i][j][0] == 0:
                dst[0] = i
                dst[1] = j
                src = np.dot(inv(H),dst)
                src = src/src[2]
                #print(dst)
                x = int(round(src[0])) % img1.shape[0]
                y = int(round(src[1])) % img1.shape[1]
                #print(x,y)
                img_tmp2[i][j] = img1[x][y]
                
    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i][j][0] != 0:
                img_tmp2[i][j] = img2[i][j]
                
    cv2.imwrite('output/2_backward.jpg', img_tmp2)          
             
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i][j][0] == 0:
                dst[0] = i
                dst[1] = j
                src = np.dot(H,dst)
                src = src/src[2]
                #print(dst)
                x = int(round(src[0])) % img2.shape[0]
                y = int(round(src[1])) % img2.shape[1]
                #print(x,y)
                img_tmp1[i][j] = img2[x][y]
                
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i][j][0] != 0:
                img_tmp1[i][j] = img1[i][j]
              
    cv2.imwrite('output/3_backward.jpg', img_tmp1)
    
def forward1(H):
    object1_point = np.load('object1.npy')
    object2_point = np.load('object2.npy')
    img = cv2.imread('images/1.jpg')
    img_tmp = np.zeros(img.shape, dtype=np.uint8)
    src = np.ones( 3 )
    dst = np.ones( 3 )
    
    mask1 = np.ones(img.shape)                              
    myROI = object1_point  # (x, y)
    cv2.fillPoly(mask1, [np.array(myROI)], 0)

    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i][j][0] == 0:
                src[0] = i
                src[1] = j
                dst = np.dot(H,src)
                dst = dst/dst[2]
                #print(dst)
                x = int(round(dst[0])) % img.shape[0]
                y = int(round(dst[1])) % img.shape[1]
                #print(x,y)
                img_tmp[x][y] = img[i][j]
    
    mask2 = np.ones(img.shape)                              
    myROI = object2_point
    cv2.fillPoly(mask2, [np.array(myROI)], 0)

    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i][j][0] == 0:
                src[0] = i
                src[1] = j
                dst = np.dot(inv(H),src)
                dst = dst/dst[2]
                #print(dst)
                x = int(round(dst[0])) % img.shape[0]
                y = int(round(dst[1])) % img.shape[1]
                #print(x,y)
                img_tmp[x][y] = img[i][j]
                
    for i in range(img_tmp.shape[0]):
        for j in range(img_tmp.shape[1]):
            if img_tmp[i][j][0] == 0:
                img_tmp[i][j] = img[i][j]
                
    cv2.imwrite('output/1_forward.jpg', img_tmp)

def forward2(H):
    object1_point = np.load('object3.npy')
    object2_point = np.load('object4.npy')
    img1 = cv2.imread('images/2.jpg')
    img2 = cv2.imread('images/3.jpg')
    img_tmp1 = np.zeros(img1.shape, dtype=np.uint8)
    img_tmp2 = np.zeros(img2.shape, dtype=np.uint8)
    src = np.ones( 3 )
    dst = np.ones( 3 )
    
    mask1 = np.ones(img1.shape)                              
    myROI = object1_point
    cv2.fillPoly(mask1, [np.array(myROI)], 0)
    
    mask2 = np.ones(img2.shape)                              
    myROI = object2_point  # (x, y)
    cv2.fillPoly(mask2, [np.array(myROI)], 0)
    
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i][j][0] == 0:
                src[0] = i
                src[1] = j
                dst = np.dot(H,src)
                dst = dst/dst[2]
                #print(dst)
                x = int(round(dst[0])) % img2.shape[0]
                y = int(round(dst[1])) % img2.shape[1]
                #print(x,y)
                img_tmp2[x][y] = img1[i][j]
                
    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i][j][0] != 0:
                img_tmp2[i][j] = img2[i][j]
    
    cv2.imwrite('output/2_forward.jpg', img_tmp2)            
             
    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i][j][0] == 0:
                src[0] = i
                src[1] = j
                dst = np.dot(inv(H),src)
                dst = dst/dst[2]
                #print(dst)
                x = int(round(dst[0])) % img1.shape[0]
                y = int(round(dst[1])) % img1.shape[1]
                #print(x,y)
                img_tmp1[x][y] = img2[i][j]
                
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i][j][0] != 0:
                img_tmp1[i][j] = img1[i][j]
                
    cv2.imwrite('output/3_forward.jpg', img_tmp1)
    
if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    
    H = project_matrix(1)
    forward1(H)
    backward1(H)
    
    H = project_matrix(2)
    forward2(H)
    backward2(H)