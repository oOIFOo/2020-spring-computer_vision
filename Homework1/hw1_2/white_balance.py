
import numpy as np

def generate_wb_mask(img, pattern, fr, fb):
    '''
    Input:
        img: H*W numpy array, RAW image
        pattern: string, 4 different Bayer patterns (GRBG, RGGB, GBRG, BGGR)
        fr: float, white balance factor of red channel
        fb: float, white balance factor of blue channel 
    Output:
        mask: H*W numpy array, white balance mask
    '''
    ########################################################################
    # TODO:                                                                #
    #   1. Create a numpy array with shape of input RAW image.             #
    #   2. According to the given Bayer pattern, fill the fr into          #
    #      correspinding red channel position and fb into correspinding    #
    #      blue channel position. Fill 1 into green channel position       #
    #      otherwise.                                                      #
    ########################################################################
    mask = np.zeros( (img.shape[0],img.shape[1]) )
    
    if (pattern == 'GRBG'):
        red = 2
        blue = 3
    if (pattern == 'RGGB'):
        red = 1
        blue = 4
    if (pattern == 'GBRG'):
        red = 3
        blue = 2
    if (pattern == 'BGGR'):
        red = 4
        blue = 1
                    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (img[i][j] == red): 
                mask[i][j] = fr
            elif (img[i][j] == blue): 
                mask[i][j] = fb
            else : 
                mask[i][j] = 1
    ########################################################################
    #                                                                      #
    #                           End of your code                           #
    #                                                                      # 
    ########################################################################
        
    return mask