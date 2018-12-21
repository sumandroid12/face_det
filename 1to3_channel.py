import sys
import numpy as np
from skimage import io
from PIL import Image
import cv2
import os


#mulitpie_dir = '/run/media/gpu/My Passport/Multi-Pie'
img_list = open(sys.argv[-1], 'r').read().split('\n')
#data_dir = '/home/gpu/suman/multipie/left_half/'

for idx, image in enumerate(img_list):
    filepath = image
    savepath = filepath.split('/')
    savepath[6] = '3ch'
    savepath = '/'.join(savepath)
    im = cv2.imread(image)
    try:
        os.makedirs('/'.join(savepath.split('/')[:-1]))
    except Exception as e:
        print(e)
        pass
    cv2.imwrite(savepath[:-3] + 'png', im)
    if idx % 100 == 0:
        print(idx)
