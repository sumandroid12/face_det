import sys
import numpy as np
import face_alignment
from skimage import io
from PIL import Image
from matplotlib import pyplot as plt

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

for image in sys.argv[1:]:
    filepath = image
    parent_dir = '/'.join(filepath.split('/')[:-2])
    save_dir = parent_dir + '/patch/'
    filename = filepath.split('/')[-1]
    name = filepath.split('/')[-1]
    print(filename)
    print(filepath)
    input = io.imread(filepath)
    preds = fa.get_landmarks(input)[-1]
    im = Image.fromarray(input)
    left_eye = (np.mean(preds[36:42, 0]), np.mean(preds[36:42, 1]))
    right_eye = (np.mean(preds[42:48, 0]), np.mean(preds[42:48, 1]))
    nose = (np.mean(preds[27:36, 0]), np.mean(preds[27:36, 1]))
    mouth = (np.mean(preds[48:68, 0]), np.mean(preds[48:68, 1]))
    bound = 20
    crop = im.crop((left_eye[0] - bound, left_eye[1] - bound, left_eye[0] + bound, left_eye[1] + bound))
    crop.save(save_dir + 'left_eye/' + name)
    crop = im.crop((right_eye[0] - bound, right_eye[1] - bound, right_eye[0] + bound, right_eye[1] + bound))
    crop.save(save_dir + 'right_eye/' + name)

    crop = im.crop((nose[0] - bound, nose[1] - bound, nose[0] + bound, nose[1] + bound))
    crop.save(save_dir + 'nose/' + name)
    crop = im.crop((mouth[0] - bound, mouth[1] - bound, mouth[0] + bound, mouth[1] + bound))
    crop.save(save_dir + 'mouth/' + name)
    print('patched ', filename)
