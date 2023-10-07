"""

"""
import os
import shutil

from glob import glob
from PIL import Image
from tqdm import tqdm

# filePath = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/3.testing/mask_split"
# fileOutput = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/newSplit/img_split"

# fileLists = glob(os.path.join(filePath, "*.png"))
# # print(len(fileLists))

# for i, fil in tqdm(enumerate(fileLists)):
#     # image = Image.open(fil)
#     # image.save(os.path.join(fileOutput, fil.split("/")[-1].split(".")[0]+"_{}.png".format(str(i))))
#     shutil.copyfile(fil, os.path.join(fileOutput, fil.split("/")[-1].split(".")[0]+"_{}.png".format(str(i))))

imgPath = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/newSplit/img_split"
# imgPath = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/img"
gtPath = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/newSplit/mask_split"
# gtPath = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/1.training_index0"

fileLists1 = sorted(glob(os.path.join(imgPath, "*.png")))
# print(len(fileLists))

fileLists2 = sorted(glob(os.path.join(gtPath, "*.png")))

# for z in fileLists1:
    # fileLists2.remove(z)
print(len(fileLists1))
print(len(fileLists2))
