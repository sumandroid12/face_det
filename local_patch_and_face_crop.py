import sys
import xml.etree.ElementTree as ET
import numpy as np
import face_alignment
from skimage import io
from PIL import Image
import os


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#mulitpie_dir = '/run/media/gpu/My Passport/Multi-Pie'
img_list = open(sys.argv[-1], 'r').read().split('\n')
#data_dir = '/home/gpu/suman/multipie/left_half/'
root_dir = '/mnt/newpartition1/chokept'

for idx, image in enumerate(img_list):
    filepath = image
    cam = image.split('/')[-2]
    frame = image.split('/')[-1][:-4]
    xfile = '/mnt/downloads/chrome/chokept/groundtruth/{}.xml'.format(cam)
    cam = xfile.split('/')[-1][:-4]
    tree = ET.parse(xfile)
    root = tree.getroot()
    try:
        id = root.find('.//frame[@number="{}"]'.format(frame)).find('person').get('id')
    except(AttributeError):
        continue

    save_dir = root_dir + '/' + '/'.join([cam, id,]) + '/'
    crop_dir = save_dir + 'crop/'
    patch_dir = save_dir + 'patch/'
    filename = filepath.split('/')[-1]
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(patch_dir + '/nose', exist_ok=True)
    os.makedirs(patch_dir + '/mouth', exist_ok=True)
    os.makedirs(patch_dir + '/left_eye', exist_ok=True)
    os.makedirs(patch_dir + '/right_eye', exist_ok=True)
    os.makedirs(save_dir + '/32x32', exist_ok=True)
    os.makedirs(save_dir + '/64x64', exist_ok=True)
    try:
        input = io.imread(filepath)
        preds = fa.get_landmarks(input)[0]
        im = Image.fromarray(input)
        min = np.min(preds, axis=0) -5
        max = np.max(preds, axis=0) +5
        crop = im.crop((min[0], min[1], max[0], max[1]))
        im128 = crop.resize((128, 128))
        im128.save(crop_dir + filename)
    except (OSError, SystemError, TypeError):
        print(filename)
        continue

    im128.resize((64, 64)).save(save_dir + '64x64/' + filename)
    im128.resize((32, 32)).save(save_dir + '32x32/' + filename)
    preds = fa.get_landmarks(np.array(im128))[0]

    left_eye = (np.mean(preds[36:42, 0]), np.mean(preds[36:42, 1]))
    right_eye = (np.mean(preds[42:48, 0]), np.mean(preds[42:48, 1]))
    nose = (np.mean(preds[27:36, 0]), np.mean(preds[27:36, 1]))
    mouth = (np.mean(preds[48:68, 0]), np.mean(preds[48:68, 1]))

    bound = 20
    crop = im128.crop((mouth[0] - bound - 4, mouth[1] - bound + 4, mouth[0] + bound + 4, mouth[1] + bound - 4))
    crop.save(patch_dir + 'mouth/' + filename)
    crop = im128.crop((left_eye[0] - bound, left_eye[1] - bound, left_eye[0] + bound, left_eye[1] + bound))
    crop.save(patch_dir + 'left_eye/' + filename)
    crop = im128.crop((right_eye[0] - bound, right_eye[1] - bound, right_eye[0] + bound, right_eye[1] + bound))
    crop.save(patch_dir + 'right_eye/' + filename)
    crop = im128.crop((nose[0] - bound, nose[1] - bound + 4, nose[0] + bound, nose[1] + bound - 4))
    crop.save(patch_dir + 'nose/' + filename)
    # print('patched ', filename)
    if (idx+1) % 10 == 0:
        print(idx)
