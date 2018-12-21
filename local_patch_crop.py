import sys
import numpy as np
import face_alignment
from skimage import io
from PIL import Image
import os


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
img_list = os.listdir(sys.argv[-1])

le = re = mo = no = np.empty((1,2))

for idx, image in enumerate(img_list):
    filepath = os.path.join(sys.argv[-1], image)
    parent_dir = '/'.join(filepath.split('/')[:-2])
    save_dir = parent_dir + '/patch/'
    filename = filepath.split('/')[-1]
    name = filepath.split('/')[-1]
    # print(filename)
    # print(filepath)
    input = io.imread(filepath)
    try:
        preds = fa.get_landmarks(input)[-1]
    except(TypeError):
        print(filename)
        continue
    im = Image.fromarray(input)
    left_eye = (np.mean(preds[36:42, 0]), np.mean(preds[36:42, 1]))
    right_eye = (np.mean(preds[42:48, 0]), np.mean(preds[42:48, 1]))
    nose = (np.mean(preds[27:36, 0]), np.mean(preds[27:36, 1]))
    mouth = (np.mean(preds[48:68, 0]), np.mean(preds[48:68, 1]))
    bound = 20

    if idx == 0:
        le = left_eye
        re = right_eye
        mo = mouth
        no = nose

    # crop = im.crop((mouth[0] - bound - 4, mouth[1] - bound + 4, mouth[0] + bound + 4, mouth[1] + bound - 4))
    # crop.save(save_dir + 'mouth/' + name)
    # crop = im.crop((left_eye[0] - bound, left_eye[1] - bound, left_eye[0] + bound, left_eye[1] + bound))
    # crop.save(save_dir + 'left_eye/' + name)
    # crop = im.crop((right_eye[0] - bound, right_eye[1] - bound, right_eye[0] + bound, right_eye[1] + bound))
    # crop.save(save_dir + 'right_eye/' + name)
    # crop = im.crop((nose[0] - bound, nose[1] - bound + 4, nose[0] + bound, nose[1] + bound - 4))
    # crop.save(save_dir + 'nose/' + name)

    le = np.vstack((le, left_eye))
    re = np.vstack((re, right_eye))
    mo = np.vstack((mo, mouth))
    no = np.vstack((no, nose))
    # print('patched ', filename)
    if idx % 100 == 0:
        print(idx)
        print(np.mean(le,axis=0), np.mean(re,axis=0), np.mean(no,axis=0), np.mean(mo,axis=0))


