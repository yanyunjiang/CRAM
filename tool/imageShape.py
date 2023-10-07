import cv2
import numpy as np


# img_path1 = "/WeaklySupervisedSemanticSegmentation/Dataset/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"
# label_path1 = "/WeaklySupervisedSemanticSegmentation/Dataset/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png"

img_path1 = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/436219-7159-48057-[1, 0, 0].png"
label_path1 = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training_mask/436219-7159-48057-[1, 0, 0].png"

img = cv2.imread(img_path1)
print("img.shape:", img.shape)


label = cv2.imread(label_path1)
print("label.shape:", label.shape)
print("unique img:", np.unique(label))
