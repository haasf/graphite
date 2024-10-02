# Default mask scoring function. High penalty for going below tr_err, otherwise
# it balances size of the mask vs. trivability

import cv2
import torch
import numpy as np

def score_fn(theta, mask, tr_err, object_size, coord = None, threshold=0.75, lbd=5, numColors=1, *args, **kargs):
    if tr_err > threshold:
        return 10000000

    # 8/5 changing lbd from 5 to 10 to weight the mask size more than the transform robustness
    # lbd = 1000000
    lbd = 10
    
    # 9/16 mask size must be < 10% of the total nBits
    # nBits = mask.sum / numColors
    # if (nBits/object_size > 0.10):
    #     return 1000000
    
    return (lbd * mask.sum() / 1) / (object_size) + tr_err
