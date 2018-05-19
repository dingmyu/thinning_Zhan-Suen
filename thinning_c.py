#%matplotlib inline
#import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from scipy import weave


def _thinningIteration(im, iter):
    I, M = im, np.zeros(im.shape, np.uint8)
    expr = """
    for (int i = 1; i < NI[0]-1; i++) {
        for (int j = 1; j < NI[1]-1; j++) {
            int p2 = I2(i-1, j);
            int p3 = I2(i-1, j+1);
            int p4 = I2(i, j+1);
            int p5 = I2(i+1, j+1);
            int p6 = I2(i+1, j);
            int p7 = I2(i+1, j-1);
            int p8 = I2(i, j-1);
            int p9 = I2(i-1, j-1);
            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
            //int m1 = iter == 0 ? 1 - ((p2 * p4 * p6 == 0) && (p4 * p6 * p8 == 0)): 0;
            //int m2 = iter == 1 ? 1 - ((p2 * p4 * p8 == 0) && (p2 * p6 * p8 == 0)): 0;
            if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
                M2(i,j) = 1;
            }
        }
    } 
    """

    weave.inline(expr, ["I", "iter", "M"])
    return (I & ~M)


def thinning(src):
    dst = src.copy()
    prev = np.zeros(src.shape[:2], np.uint8)
    diff = None

    while True:
        dst = _thinningIteration(dst, 0)
        dst = _thinningIteration(dst, 1)
        diff = np.absolute(dst - prev)
        prev = dst.copy()
        if np.sum(diff) == 0:
            break
    dst[0,:] = 0
    dst[-1,:]=0
    dst[:,0] = 0
    dst[:,-1] = 0
    return dst

 
gt_name = '/mnt/lustre/drive_data/adas-video/record/GMXL1207/Gt_png/GMXL_Img/20171103/Random/Sunny/Night/20170927204720.MP4/00000700.png'
img = cv2.imread(gt_name,-1)
img = cv2.resize(img,(209,177))
img = np.clip(img,0,1)
thinning_img = thinning(img)
image = Image.fromarray(thinning_img.astype('uint8')*255)
image.save('thinning.png')