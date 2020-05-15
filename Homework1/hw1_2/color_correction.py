import cv2
import skimage
import numpy as np

def BGR2RGB(img):
    '''
    Input:
        img: H*W*3, input BGR image
    Output:
        output: H*W*3, output RGB image
    '''
    b, g, r = cv2.split(img)
    output = cv2.merge([r, g, b])
    return output

def RGB2BGR(img):
    '''
    Input:
        img: H*W*3, input RGB image
    Output:
        output: H*W*3, output BGR image
    '''
    r, g, b = cv2.split(img)
    output = cv2.merge([b, g, r])
    return output

def RGB2XYZ(img):
    '''
    Input:
        img: H*W*3, input RGB image
    Output:
        output: H*W*3, output CIE XYZ image
    '''
    output = skimage.color.rgb2xyz(img)
    return output

def XYZ2RGB(img):
    '''
    Input:
        img: H*W*3, input CIE XYZ image
    Output:
        output: H*W*3, output RGB image
    '''
    output = skimage.color.xyz2rgb(img)
    return output


def color_correction(img, ccm):
    '''
    Input:
        img: H*W*3 numpy array, input image
        ccm: 3*3 numpy array, color correction matrix 
    Output:
        output: H*W*3 numpy array, output image after color correction
    '''
    ########################################################################
    # TODO:                                                                #
    #   Following the p.22 of hw1_tutorial.pdf to get P as output.         #
    #                                                                      #
    ########################################################################
    matrix_O = np.zeros( (img.shape[0]*img.shape[1], img.shape[2]) )
    matrix_P = np.zeros( (img.shape[0]*img.shape[1], img.shape[2]) )
    output = np.zeros( (img.shape[0],img.shape[1],img.shape[2]) )
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for z in range(img.shape[2]):
                matrix_O[x*img.shape[1] + y][z] = img[x][y][z]
            
    for x in range(matrix_P.shape[0]):
        matrix_P[x][0] = matrix_O[x][0]*ccm[0][0] + matrix_O[x][1]*ccm[1][0] + matrix_O[x][2]*ccm[2][0]
        matrix_P[x][1] = matrix_O[x][0]*ccm[0][1] + matrix_O[x][1]*ccm[1][1] + matrix_O[x][2]*ccm[2][1]
        matrix_P[x][2] = matrix_O[x][0]*ccm[0][2] + matrix_O[x][1]*ccm[1][2] + matrix_O[x][2]*ccm[2][2]
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for z in range(img.shape[2]):
                output[x][y][z] = matrix_P[x*img.shape[1] + y][z]
    
   # output = img.dot(ccm)
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
    #### Prevent the value larger than 1 or less than 0
    output = np.clip(output, 0, 1)
    return output
