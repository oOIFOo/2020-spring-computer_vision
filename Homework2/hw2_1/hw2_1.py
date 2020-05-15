from scipy import linalg
import os
import cv2
import numpy as np
from numpy.linalg import inv

def project_matrix(idx):
    two_d_point = np.load('Point2D_'+str(idx)+'.npy')
    three_d_point = np.loadtxt('Point3D.txt')
    tmp = np.ones( (13,12) )
    P = np.ones( (3,4) )
    a = np.zeros( (13,1) )
    a[12][0] = 1
    
    for i in range(12):
        j = int(i/2)*6
        if i % 2 == 0:
            for x in range(3):
                tmp[i][x] = three_d_point[j][x]
                
            for x in range(4):
                tmp[i][x+4] = 0
                
            for x in range(3):
                tmp[i][x+8] = -1*two_d_point[j][0]*three_d_point[j][x]
                
            tmp[i][11] = -two_d_point[j][0]
            
        elif i % 2 == 1:
            for x in range(4):
                tmp[i][x] = 0
                
            for x in range(3):
                tmp[i][x+4] = three_d_point[j][x]   
            
            for x in range(3):
                tmp[i][x+8] = -1*two_d_point[j][1]*three_d_point[j][x]
                
            tmp[i][11] = -two_d_point[j][1]
    
    #print(tmp)
    Z = np.dot(np.linalg.pinv(tmp),a)
    #print(Z)
    for x in range(12):
       P[int(x/4)][x%4] =  Z[x][0]
    
    print("projection matrix = ")
    print(P)
    return P

def RQ_decompose(P):
    M = np.zeros( (3,3) )
    P4 = np.zeros( (3,1) )
    for i in range(3):
        for j in range(3):
            M[i][j] = P[i][j]
    
    #####Negate P if P3×3 has negative determinant
    if np.linalg.det(M) < 0:
        M = -M
        P = -P
        
    for i in range(3):
        P4[i][0] = P[i][3]
    
    #print(M)
    #print(P4)
    #####K=RD,R=DQ where M=RQ    
    tmp = linalg.rq(M)
    _R = tmp[0]
    _Q = tmp[1]
    #print(_R)
    #print(_Q)

    D = np.zeros( (3,3) )
    D[0][0] = np.sign(_R[0][0])
    D[1][1] = np.sign(_R[1][1])
    D[2][2] = np.sign(_R[2][2])
    #print(D)
    
    K = np.dot(_R,D)
    R = np.dot(D,_Q)
    #print(K)
    #print(R)
    
    #####T=inv(K)*P4
    T = np.dot(inv(K),P4)
    
    #####Scale K such that K3,3=1
    K = K/K[-1][-1]
    
    print("K = ")
    print(K)
    print("R = ")
    print(R)
    print("T = ")
    print(T)
    tmp_2D = reproject(K,R,T)
    
    return tmp_2D

def reproject(K,R,T):
    tmp_2D = np.ones( (36,3) )
    real_2D = np.ones( (36,3) )
    real_3D = np.ones( (36,4) )
    RT = np.zeros( (3,4) )
    error = 0
    
    two_d_point = np.load('Point2D_'+str(idx)+'.npy')
    three_d_point = np.loadtxt('Point3D.txt')
    
    for i in range(two_d_point.shape[0]):
        for j in range(two_d_point.shape[1]):
            real_2D[i][j] = two_d_point[i][j]
            
    for i in range(three_d_point.shape[0]):
        for j in range(three_d_point.shape[1]):
            real_3D[i][j] = three_d_point[i][j]
            
    for i in range(3):
        RT[i][3] = T[i]
        for j in range(3):
            RT[i][j] = R[i][j]
            
    #print(RT)        
    KRT = np.dot(K,RT)     
    #print(KRT)
    
    for x in range(two_d_point.shape[0]):     
        tmp_2D[x] = np.dot(KRT,real_3D[x])
        tmp_2D[x] = tmp_2D[x]/tmp_2D[x][2]
    
    for x in range(three_d_point.shape[0]):
        #print(tmp_2D[x])
        #print(real_2D[x])
        error = error + ((tmp_2D[x] - real_2D[x])**2).mean(axis=0)/36
    
    print("root-mean-squared errors = ")    
    print(error)
    return tmp_2D

def point_img(tmp_2D,idx):
    two_d_point = np.load('Point2D_'+str(idx)+'.npy')
    img_folder_path = 'data'
    img_folder_save_path = 'outputs'
    img_path = os.path.join(img_folder_path, 'chessboard_'+str(idx)+'.jpg')
    img = cv2.imread(img_path) 
    
    for i in range(36):
        cv2.circle(img, (int(two_d_point[i][0]), int(two_d_point[i][1])), 5, (0, 255, 255), -1)
    for i in range(36):
        cv2.circle(img, (int(tmp_2D[i][0]), int(tmp_2D[i][1])), 2, (0, 0, 255), -1)
     
    save_name = 'chessboard_'+str(idx)+'.jpg'
    save_img_path = os.path.join(img_folder_save_path, save_name)
    cv2.imwrite(save_img_path, img)
    
if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    idx = 1
    P = project_matrix(idx)
    tmp_2D = RQ_decompose(P)
    point_img(tmp_2D,idx)
    
    idx = 2
    P = project_matrix(idx)
    tmp_2D = RQ_decompose(P)
    point_img(tmp_2D,idx)
    