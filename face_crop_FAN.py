import sys
import os
import face_alignment
from skimage import io
from PIL import Image

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

img_dir = os.listdir(sys.argv[-1])

for idx, image in enumerate(img_dir):
    filepath = os.path.join(sys.argv[-1], image)
    parent_dir = '/'.join(filepath.split('/')[:-2])
    save_dir = parent_dir + '/'
    filename = filepath.split('/')[-1]
    name = filepath.split('/')[-1]
    print(filename)
    # print(filepath)
    input = io.imread(filepath)
    preds = fa.get_landmarks(input)[-1]
    im = Image.fromarray(input)
    crop = im.crop((preds[0,0]-15, preds[24,1] -30 , preds[16,0]+15 , preds[8,1] ))
    crop = crop.resize((128, 128))
    try:
        crop.save(save_dir + 'crop/' + name)
    except(SystemError):
        continue
    if idx % 100 == 0:
        print(idx)

