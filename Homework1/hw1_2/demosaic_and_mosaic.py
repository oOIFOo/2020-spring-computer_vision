
import numpy as np

from demosaic_2004 import demosaicing_CFA_Bayer_Malvar2004

def mosaic(img, pattern):
    '''
    Input:
        img: H*W*3 numpy array, input image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W numpy array, output image after mosaic.
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create the H*W output numpy array.                              #   
    #   2. Discard other two channels from input 3-channel image according #
    #      to given Bayer pattern.                                         #
    #                                                                      #
    #   e.g. If Bayer pattern now is BGGR, for the upper left pixel from   #
    #        each four-pixel square, we should discard R and G channel     #
    #        and keep B channel of input image.                            #     
    #        (since upper left pixel is B in BGGR bayer pattern)           #
    ########################################################################
    output = np.zeros( (img.shape[0],img.shape[1]) )

    if (pattern == 'GRBG'):
        channel = [1, 0, 2, 1]
    if (pattern == 'RGGB'):
        channel = [0, 1, 1, 2]
    if (pattern == 'GBRG'):
        channel = [1, 2, 0, 1]
    if (pattern == 'BGGR'):
        channel = [2, 1, 1, 0]
        
    for i in range(img.shape[0]):
        for j in range(output.shape[1]):
            if((i % 2) == 0):
                if((j % 2) == 0):
                    output[i][j] = img[i][j][channel[0]]
                elif((j % 2) == 1):
                    output[i][j] = img[i][j][channel[1]]
            if((i % 2) == 1):
                if((j % 2) == 0):
                    output[i][j] = img[i][j][channel[2]]
                elif((j % 2) == 1):
                    output[i][j] = img[i][j][channel[3]]
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################

    return output


def demosaic(img, pattern):
    '''
    Input:
        img: H*W numpy array, input RAW image.
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
    Output:
        output: H*W*3 numpy array, output de-mosaic image.
    '''
    #### Using Python colour_demosaicing library
    #### You can write your own version, too
    output = demosaicing_CFA_Bayer_Malvar2004(img, pattern)
    output = np.clip(output, 0, 1)

    return output

