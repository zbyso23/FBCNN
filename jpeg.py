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

# PyTorch simple dataset + model + train
# https://gist.github.com/subhadarship/54f6d320ba34afe80766ca89a1ceb448

# CV2 Extract channel
# https://stackoverflow.com/questions/54069274/how-to-extract-green-channel-using-opencv-in-python
# CV2 Resize
# https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/#gsc.tab=0

def main():
    JPEG_QUALITY = 90
    BOX_SIZE = 8
    n_channels = 3

    folder_in = 'testsets/'
    batch = 'Real'
    filename = '3'
    ext = 'jpg'

    folder_out = 'custom/dataset/'
    
    os.mkdir(folder_out + batch)

    img_in = folder_in + '/' + batch + '/' + filename + ext
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





    # img_L = util.imread_jpeg_uint(img_in, n_channels=n_channels)
    # if img_L is None:
    #     return
    # util.imsave(img_L, img_out)
    # util.jpegsave(img_L, img_out, JPEG_QUALITY)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()