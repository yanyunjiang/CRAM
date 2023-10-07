from PIL import Image
import numpy as np


cam_path = "/WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226_3cam/436219-35356-55497-[1, 1, 0].png"

cam_path = "/WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226_3cam/5.png"

cam = Image.open(cam_path)

print(np.array(cam).shape)

print(np.array(cam)[:,100])