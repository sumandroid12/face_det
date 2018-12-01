import sys
import numpy as np
import face_alignment
from skimage import io
from PIL import Image
import os


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

mulitpie_dir = '/run/media/gpu/My Passport/Multi-Pie'
img_list = open(sys.argv[-1], 'r').read().split('\n')
data_dir = '/home/gpu/suman/multipie/left_half/'

for idx, image in enumerate(img_list):
    filepath = mulitpie_dir + image[1:]
    save_dir = data_dir + image.split('/')[-1].split('_')[-2] + '/'
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
        preds = fa.get_landmarks(input)[-1]
        im = Image.fromarray(input)
        crop = im.crop((preds[0, 0] - 15, preds[24, 1] - 30, preds[16, 0] + 15, preds[8, 1]))
        crop = crop.resize((128, 128))
        im = crop
        preds = fa.get_landmarks(np.array(crop))[-1]
        crop.save(crop_dir + filename)
    except (OSError, SystemError, TypeError):
        print(filename)
        continue

    crop.resize((64, 64)).save(save_dir + '64x64/' + filename)
    crop.resize((32, 32)).save(save_dir + '32x32/' + filename)

    left_eye = (np.mean(preds[36:42, 0]), np.mean(preds[36:42, 1]))
    right_eye = (np.mean(preds[42:48, 0]), np.mean(preds[42:48, 1]))
    nose = (np.mean(preds[27:36, 0]), np.mean(preds[27:36, 1]))
    mouth = (np.mean(preds[48:68, 0]), np.mean(preds[48:68, 1]))

    bound = 20
    crop = im.crop((mouth[0] - bound - 4, mouth[1] - bound + 4, mouth[0] + bound + 4, mouth[1] + bound - 4))
    crop.save(patch_dir + 'mouth/' + filename)
    crop = im.crop((left_eye[0] - bound, left_eye[1] - bound, left_eye[0] + bound, left_eye[1] + bound))
    crop.save(patch_dir + 'left_eye/' + filename)
    crop = im.crop((right_eye[0] - bound, right_eye[1] - bound, right_eye[0] + bound, right_eye[1] + bound))
    crop.save(patch_dir + 'right_eye/' + filename)

    crop = im.crop((nose[0] - bound, nose[1] - bound + 4, nose[0] + bound, nose[1] + bound - 4))
    crop.save(patch_dir + 'nose/' + filename)
    # print('patched ', filename)
    if (idx+1) % 10 == 0:
        print(idx)
