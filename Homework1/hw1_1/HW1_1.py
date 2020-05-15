import math
import os
import cv2
import scipy.io
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def gaussian_smooth(img, size):
    pred = np.zeros( (img.shape[0], img.shape[1], img.shape[2]) )
    tmp = np.zeros( (img.shape[0], img.shape[1]) )
    
    kernel_size = size
    sigma = 5
    x, y = np.mgrid[int((-kernel_size)/2):kernel_size - 2, int((-kernel_size)/2):kernel_size - 2]
    #print (x)
    gaussian_kernel = np.exp(-((x**2+y**2)/2*sigma**2))
    
    #Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()   
    
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    
    i = 0
    for (chan, color) in zip(chans, colors):
        tmp = signal.convolve2d(chan, gaussian_kernel, boundary='symm', mode='same') #卷積
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                pred[x][y][i] = tmp[x][y]     
        i += 1
    
    save_name = str(idx+1)+'_gaussian, kernel = '+str(kernel_size)+'.jpg'
    save_img_path = os.path.join(img_folder_save_path, save_name)
    cv2.imwrite(save_img_path, pred)
    
def sobel_edge_detection(img):
    print (" ")
    
def structure_tensor():
    print (" ")
    
def nms():
    print (" ")
    
def process_one_image(img_path):
    img = cv2.imread(img_path)
    for i in range(2):
        gaussian_smooth(img, (i+1)*5)
        
    sobel_edge_detection(img)
    
if __name__ == '__main__':
    img_num = 2
    img_folder_path = 'images'
    img_folder_save_path = 'results'
    
    for idx in range(img_num):
        img_path = os.path.join(img_folder_path, str(idx+1)+'.jpg')
        print(img_path)
        process_one_image(img_path)

        print('')