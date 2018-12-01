import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

input = io.imread('/home/suman/datasets/lfw/lfw/Abdullah_Gul/Abdullah_Gul_0007.jpg')
preds = fa.get_landmarks(input)
