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
    BOX_SIZE = 8
    n_channels = 3
    img_in = 'testsets/Real/3.jpg'
    img_out = 'testsets/Real/3_out.png'

    img_rgb = cv2.imread(img_in)
    height, width, channels = img_rgb.shape
    
    scale_percent = 200 # percent of original size
    width = int(img_rgb.shape[1] * scale_percent / 100)
    height = int(img_rgb.shape[0] * scale_percent / 100)
    dim = (width, height)
    img_rgb = cv2.resize(img_rgb, dim, interpolation = cv2.INTER_AREA)


    img_ycrcb = img_rgb.copy()
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

    # Cycle per 8x8 boxes
    # print(height, width, channels)
    # for x in range(0, width, BOX_SIZE):
    #     for y in range(0, height, BOX_SIZE):
    #         print(x, y)
    #         crop_img_rgb = img_rgb[y:y+BOX_SIZE, x:x+BOX_SIZE]
    #         crop_img_ycrcb = img_ycrcb[y:y+BOX_SIZE, x:x+BOX_SIZE]



    # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_YCrCb2RGB)
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)

    # Check if is Box in one color
    # x = 0; y = 0
    # crop_img_rgb = img_rgb[y:y+BOX_SIZE, x:x+BOX_SIZE]
    # crop_flatten = list(chain.from_iterable(list(chain.from_iterable(crop_img_rgb))))
    # is_one_color = crop_flatten.count(crop_flatten[0]) == len(crop_flatten)
    # # pixel = np.argwhere(crop_flatten == 223)
    # print(is_one_color)
    # print(crop_flatten)

    # Encode RGB image data to JPEG and decode back, after convert to YCbCr and show per channel using detective fn
    # quality_factor = 5
    # noise_level = (100-quality_factor)/100.0
    # result, encimg = cv2.imencode('.jpg', img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    # img_rgb_jpeg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    # cv2.imshow('img jpeg', img_rgb_jpeg)
    # cv2.waitKey(0)
    # img_ycrcb_jpeg = cv2.cvtColor(img_rgb_jpeg, cv2.COLOR_RGB2YCrCb)
    # detective(img_ycrcb_jpeg, ['Y', 'Cb', 'Cr'], 'jpeg')

    # CMYK Testing JPEG decoded data
    # for i in ['C', 'M', 'Y', 'K']:
    #     img_cmy = tocmyk(img_rgb_jpeg, i)
    #     cv2.imshow('img_cmy '+i, img_cmy)
    #     cv2.waitKey(0)

    # Crop image to BOX_SIZE and check if is only one color used
    # x = 0; y = 0
    # crop_img_rgb = img_rgb[y:y+BOX_SIZE, x:x+BOX_SIZE]
    # crop_flatten = list(chain.from_iterable(list(chain.from_iterable(crop_img_rgb))))
    # is_one_color = crop_flatten.count(crop_flatten[0]) == len(crop_flatten)
    # print(is_one_color)

    # Show Green channel from RGB by replacing R and B by zeros
    # Y_img = crop_img.copy()
    # Y_img[:,:,0] = 255
    # Y_img[:,:,2] = 255
    # cv2.imshow('G-RGB', Y_img)
    # cv2.waitKey(0)

    # Show Y Cr and Cb channels separate
    # y, cr, cb = cv2.split(img_ycrcb)
    # cv2.imshow('Y', y)
    # cv2.waitKey(0)
    # cv2.imshow('Cr', cr)
    # cv2.waitKey(0)
    # cv2.imshow('Cb', cb)
    # cv2.waitKey(0)

    # img_L = util.imread_jpeg_uint(img_in, n_channels=n_channels)
    # if img_L is None:
    #     return
    # util.imsave(img_L, img_out)
    # util.jpegsave(img_L, img_out, JPEG_QUALITY)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()