"""
生成不同通道的feature map热图
"""

import os
import numpy as np

import cv2
from PIL import Image

image_name = "/WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226_wsss4luad_numpy/1124863-24347-30642-[0, 0, 1].npy"
output_ = np.load(image_name)

image_name = image_name.split("/")[-1].split(".")[0]

print(output_.shape) # [3, h, w]

for i in range(output_.shape[0]):

    # gray_img = Image.fromarray((output_[i]*100).astype(np.uint8))
    # gray_img.save(os.path.join("/WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226_wsss4luad_numpy", "{}.png".format(image_name + "_gray_" +str(i))))

    im_color = cv2.applyColorMap((output_[i]*100).astype(np.uint8), 4)
    new_img = Image.fromarray(im_color)
    new_img.save(os.path.join("/WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226_wsss4luad_numpy", "{}.png".format(image_name + "_color4_" +str(i))))
# quit(0)