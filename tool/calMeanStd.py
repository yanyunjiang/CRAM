import numpy as np
import cv2
import random
import os
from tqdm import tqdm
# calculate means and std
train_txt_path = './train_val_list.txt'

CNum = 10000   # 挑选多少图片进行计算



os.environ["LANDMARK_IMAGE_PATH"] = '/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/img'
data_root = os.environ['LANDMARK_IMAGE_PATH']
image_root = os.path.join(data_root, '')        #'db', 'coco', 'images')
 
from os import listdir
from os.path import isfile, join
 
import pathlib
# print(pathlib.Path('yourPath.example').suffix) # this will give result  '.example'

img_h, img_w = 128, 128
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

onlyfiles = [f for f in listdir(image_root) if isfile(join(image_root, f)) ]
my_imgfiles = []
for i  in range(len(onlyfiles)):
    if(pathlib.Path(onlyfiles[i]).suffix=='.png') or (pathlib.Path(onlyfiles[i]).suffix=='.jpg') or (pathlib.Path(onlyfiles[i]).suffix=='.jpeg'):
        my_imgfiles.append(onlyfiles[i])


for i in tqdm(range(len(my_imgfiles))):
    img_name = os.path.join(image_root, my_imgfiles[i])  

    # img_path = os.path.join('./train', lines[i].rstrip().split()[0])

    img = cv2.imread(img_name)
    img = cv2.resize(img, (img_h, img_w))
    img = img[:, :, :, np.newaxis]

    imgs = np.concatenate((imgs, img), axis=3)
#     print(i)

imgs = imgs.astype(np.float32)#/255.


for i in tqdm(range(3)):
    pixels = imgs[:,:,i,:].ravel() # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse() # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))