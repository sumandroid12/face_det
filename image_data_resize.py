import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
from PIL import Image

# root path depends on your computer
resize_size = 64
root = '/home/gpu/suman/multipie/test/pose/middle/normal/crop/'
save_root = '/home/gpu/suman/multipie/test/pose/middle/normal/{}x{}/'.format(resize_size, resize_size)


# if not os.path.isdir(save_root):
#     os.mkdir(save_root)
# if not os.path.isdir(save_root + 'celebA'):
#     os.mkdir(save_root + 'celebA')

# ten_percent = len(img_list) // 10
img_list = os.listdir(root)

for i in range(len(img_list)):
    img = Image.open(root + img_list[i])
    img = img.resize((resize_size, resize_size))
    img.save(save_root + img_list[i])

    if (i % 1000) == 0:
        print('%d images complete' % i)