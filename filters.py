"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    temp_m = np.zeros((Hi+Hk-1, Wi+Wk-1))     # 所得为 full 矩阵
    for i in range(Hi+Hk-1):
        for j in range(Wi+Wk-1):
            temp = 0
            # 通常来说，卷积核的尺寸远小于图片尺寸，同时卷积满足交换律，为了加快运算，可用h*f 代替 f*h 进行计算
            for m in range(Hk):
                for n in range(Wk):
                    if ((i-m)>=0 and (i-m)<Hi and (j-n)>=0 and (j-n)<Wi):
                        temp += image[i-m][j-n] * kernel[m][n]
            temp_m[i][j] = temp
    # 截取出 same 矩阵 （输出尺寸同输入）
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = temp_m[int(i+(Hk-1)/2)][int(j+(Wk-1)/2)]

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    pass
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height:pad_height+H, pad_width:pad_width+W] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_height = Hk // 2
    pad_width = Wk // 2
    image_padding = zero_pad(image, pad_height, pad_width)
    kernel_flip = np.flip(np.flip(kernel, 0), 1)

    for i in range(Hi):
        for j in range(Wi):            
            out[i][j] = np.sum(np.multiply(kernel_flip, image_padding[i:(i+Hk), j:(j+Wk)]))    
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pass
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    
    pad_height = Hg // 2
    pad_width = Wg // 2
    f_padding = zero_pad(f, pad_height, pad_width)  

    for i in range(Hf):
        for j in range(Wf):            
            out[i][j] = np.sum(np.multiply(g, f_padding[i:(i+Hg), j:(j+Wg)]))    
   
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pass    
    Hg, Wg = g.shape
    mean = np.mean(g)    
    for i in range(Hg):
        for j in range(Wg):
            g[i][j] = g[i][j]-mean
    out = cross_correlation(f,g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pass
    Hg, Wg = g.shape 
    gmean = np.mean(g)
    stdg=np.std(g)
    for m in range(Hg):
        for n in range(Wg):
            g[m][n] = (g[m][n]-gmean)/stdg

    Hf, Wf = f.shape    
    out = np.zeros((Hf, Wf))
    pad_height = Hg // 2
    pad_width = Wg // 2
    f_padding1 = zero_pad(f, pad_height, pad_width)
    f_padding2 = f_padding1
    
    for i in range(Hf):
        for j in range(Wf):
            temp1 = f_padding1[i:(i+Hg), j:(j+Wg)]
            meanf=np.mean(temp1)
            stdf=np.std(temp1)

            for m in range(Hg):
                for n in range(Wg):
                    temp1[m][n] = (temp1[m][n]-meanf)/stdf

            out [i][j] = cross_correlation(g,temp1).sum()
    ### END YOUR CODE

    return out
