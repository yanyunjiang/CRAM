import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image

palette = []
for i in range(256):
    palette.extend((i, i, i))
palette[:3 * 4] = np.array([[0, 64, 128],
                            [64, 128, 0],
                            [243, 152, 0],
                            [255, 255, 2555]
                            ], dtype='uint8').flatten()

file_path = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/1.training_mask/"
# file_name = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/2.validation/mask/00.png"

file_list = sorted(glob(file_path+"*.png"))
out_file_path = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/1.training_index0/"

if not os.path.exists(out_file_path):
    os.mkdir(out_file_path)

for fil in tqdm(file_list):
    img = cv2.imread(fil)
    img = img[:, :, ::-1] # BGR ==> RGB
    # arr = np.unique(img)
    # print("arr:", arr)
    img_0 = np.zeros_like(img)[:,:,0]
    (x, y,_) = img.shape
    
    for i in range(x):
        for j in range(y):
            l, m, n = img[i, j, :]
            # print("l:", l, "m:", m, "n:", n)
            if l == 0 and m == 0 and n == 0:
                # print("1")
                img_0[i, j] = 0
                
            elif l == 128 and m == 0 and n == 0:
                # print("2")
                img_0[i, j] = 1
                # img_0[i, j, 1] = 0
                # img_0[i, j, 2] = 0
            elif l == 0 and m == 128 and n == 0:
                # print("3")
                img_0[i, j] = 2
                # img_0[i, j, 1] = 128
                # img_0[i, j, 2] = 0  
            elif l == 128 and m == 128 and n == 0:
                # print("4")
                img_0[i, j] = 3
                # img_0[i, j, 1] = 128
                # img_0[i, j, 2] = 0
    img_0 = np.array(img_0,  dtype=np.uint8)
    # print(np.unique(img_0 - 1))
    img_ = (img_0 - 1) +  ((img_0 - 1) == 255) * 4
    img_ = Image.fromarray(np.uint8(img_))
    # img_0 = cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(out_file_path+fil.split("/")[-1], img_)

    img_.putpalette(palette)
    img_.save(out_file_path+fil.split("/")[-1])

    


#            train           val
# 背景 白色：[0, 0, 0] ==> [255, 255, 255]
# 肿瘤 蓝色：[128, 0 , 0] ==>   [0, 64, 128]
# 间质 绿色：[0, 128, 0] ==> [64, 128, 0]
# 正常 橘黄色: [128, 128, 0] ==> [243, 152, 0]