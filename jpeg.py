import cv2
import os.path
import logging
import numpy as np
from datetime import datetime
from collections import OrderedDict
import torch
import cv2
from utils import utils_logger
from utils import utils_image as util
import requests
from pathlib import Path
import time
from itertools import chain

# PyTorch simple dataset + model + train
# https://gist.github.com/subhadarship/54f6d320ba34afe80766ca89a1ceb448

# GIST Train epoch
# https://gist.github.com/kriventsov/4f8b42cfc248aae6c94a25462fab833c
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

# CV2 Extract channel
# https://stackoverflow.com/questions/54069274/how-to-extract-green-channel-using-opencv-in-python
# CV2 Resize
# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/#gsc.tab=0

def tocmyk(img, channel=None):
    img_data = img.astype(np.float32)/255.
    
    K = 1 - np.max(img_data, axis=2) # Calculate K as (1 - whatever is biggest out of R, G, B)
    C = (1-img_data[...,2] - K)/(1-K)
    M = (1-img_data[...,1] - K)/(1-K)
    Y = (1-img_data[...,0] - K)/(1-K)
    if channel is None:
        CMY = (np.dstack((C,M,Y)) * 255).astype(np.uint8)    
    elif channel == 'C':
        CMY = (np.dstack((C,C,C)) * 255).astype(np.uint8)
    elif channel == 'M':
        CMY = (np.dstack((M,M,M)) * 255).astype(np.uint8)
    elif channel == 'Y':
        CMY = (np.dstack((Y,Y,Y)) * 255).astype(np.uint8)
    else:
        CMY = (np.dstack((K,K,K)) * 255).astype(np.uint8)

    # Combine 3 or 4 channels into single image and re-scale back up to uint8
    # CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)
    # CMY = (np.dstack((C,M,Y)) * 255).astype(np.uint8)
    # return CMYK
    return CMY


def detective(img, channels, name='img'):
    index = 0
    for i in channels:
        img_part = img.copy()
        for x in range(3):
            print(x)
            if index != x:
                img_part[:,:,x] = img_part[:,:,index]
       
        cv2.imshow(name + ' ' + i, img_part)
        cv2.waitKey(0)
        index = index + 1



def main():
    JPEG_QUALITY = 90
    BOX_SIZE = 2
    n_channels = 3

    # folder_in = 'testsets'
    # batch = 'Real'
    # filename = '3'
    ext = 'jpg'


    folder_in = 'datasets/custom'
    batch = 'anime'
    filename = 'raw/019'
    ext = 'png'

    folder_out = 'custom/dataset/'
    
    # os.mkdir(folder_out + batch)

    img_in = folder_in + '/' + batch + '/' + filename + '.' + ext
    img_out = 'testsets/Real/3_out.png'

    print(img_in)
    img_rgb = cv2.imread(img_in)
    height, width, channels = img_rgb.shape
    
    scale_percent = 100 # percent of original size
    width = int(img_rgb.shape[1] * scale_percent / 100)
    height = int(img_rgb.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_rgb = cv2.resize(img_rgb, dim, interpolation = cv2.INTER_AREA)
    # detective(img_rgb, ['R', 'G', 'B'])

    # cv2.imwrite('datasets/custom/anime/raw/003.png', img_rgb)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    detective(img_ycrcb, ['Y', 'Cb', 'Cr'])

    quality_factor = 5
    noise_level = (100-quality_factor)/100.0
    result, encimg = cv2.imencode('.jpg', img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img_rgb_jpeg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    # detective(img_rgb_jpeg, ['R', 'G', 'B'])

    img_ycrcb_jpeg = cv2.cvtColor(img_rgb_jpeg, cv2.COLOR_RGB2YCrCb)
    detective(img_ycrcb_jpeg, ['Y', 'Cb', 'Cr'], 'jpeg')

    img_hls_jpeg = cv2.cvtColor(img_rgb_jpeg, cv2.COLOR_RGB2HLS)
    detective(img_hls_jpeg, ['H', 'L', 'S'], 'hls jpeg')

    img_hsv_jpeg = cv2.cvtColor(img_rgb_jpeg, cv2.COLOR_RGB2HSV)
    detective(img_hsv_jpeg, ['H', 'S', 'V'], 'hsv jpeg')

    # cv2.imshow('img jpeg', img_rgb_jpeg)
    # cv2.waitKey(0)

    # img_yuv_jpeg = cv2.cvtColor(img_rgb_jpeg, cv2.COLOR_RGB2YUV)
    # detective(img_yuv_jpeg, ['Y', 'U', 'V'], 'yuv_jpeg')

    # HLS Testing
    # img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    # detective(img_hls, ['H', 'L', 'S'])
    




    # CMYK Testing
    # for i in ['C', 'M', 'Y', 'K']:
    #     img_cmy = tocmyk(img_rgb_jpeg, i)
    #     cv2.imshow('img_cmy '+i, img_cmy)
    #     cv2.waitKey(0)

    # cv2.imshow('img_rgb', img_rgb)
    # cv2.waitKey(0)
    # cv2.imshow('img_ycrcb', img_ycrcb)
    # cv2.waitKey(0)
    # cv2.imshow('img_yuv', img_yuv)
    # cv2.waitKey(0)
    # cv2.imshow('img_hls', img_hls)
    # cv2.waitKey(0)



    # Cycle per 8x8 boxes
    # print(height, width, channels)
    # for x in range(0, width, BOX_SIZE):
    #     for y in range(0, height, BOX_SIZE):
    #         print(x, y)
    #         crop_img_rgb = img_rgb[y:y+BOX_SIZE, x:x+BOX_SIZE]
    #         crop_img_ycrcb = img_ycrcb[y:y+BOX_SIZE, x:x+BOX_SIZE]


    # Check if is Box in one color
    x = 0; y = 0
    crop_img_rgb = img_rgb[y:y+BOX_SIZE, x:x+BOX_SIZE]
    crop_flatten = list(chain.from_iterable(list(chain.from_iterable(crop_img_rgb))))
    is_one_color = crop_flatten.count(crop_flatten[0]) == len(crop_flatten)
    print(is_one_color)
    print(crop_flatten)


    # img_L = util.imread_jpeg_uint(img_in, n_channels=n_channels)
    # if img_L is None:
    #     return
    # util.imsave(img_L, img_out)
    # util.jpegsave(img_L, img_out, JPEG_QUALITY)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()