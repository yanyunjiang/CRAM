import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm


file_path = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training_mask/"
# file_name = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/2.validation/mask/00.png"

file_list = sorted(glob(file_path+"*.png"))
out_file_path = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training_index/"
for fil in tqdm(file_list):
    img = cv2.imread(fil)
    img = img[:, :, ::-1] # BGR ==> RGB
    # arr = np.unique(img)
    # print("arr:", arr)
    img_0 = np.zeros_like(img)
    (x, y, _) = img.shape
    
    for i in range(x):
        for j in range(y):
            l, m, n = img[i, j, :]
            # print("l:", l, "m:", m, "n:", n)
            if l == 255 and m == 255 and n == 255:
                # print("1")
                img_0[i, j, 0] = 0
                img_0[i, j, 1] = 0
                img_0[i, j, 2] = 0
            elif l == 0 and m == 64 and n == 128:
                # print("2")
                img_0[i, j, 0] = 128
                img_0[i, j, 1] = 0
                img_0[i, j, 2] = 0
            elif l == 64 and m == 128 and n == 0:
                # print("3")
                img_0[i, j, 0] = 0
                img_0[i, j, 1] = 128
                img_0[i, j, 2] = 0  
            elif l == 243 and m == 152 and n == 0:
                # print("4")
                img_0[i, j, 0] = 128
                img_0[i, j, 1] = 128
                img_0[i, j, 2] = 0
    img_0 = np.array(img_0,  dtype=np.uint8)
    img_0 = cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_file_path+fil.split("/")[-1], img_0)

    


#            train           val
# 背景 白色：[0, 0, 0] ==> [255, 255, 255]
# 肿瘤 蓝色：[128, 0 , 0] ==>   [0, 64, 128]
# 间质 绿色：[0, 128, 0] ==> [64, 128, 0]
# 正常 橘黄色: [128, 128, 0] ==> [243, 152, 0]