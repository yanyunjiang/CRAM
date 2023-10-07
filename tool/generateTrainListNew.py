import os
from glob import glob
import random

luadFilesPaths = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/newSplit/img_split"

fileLists = sorted(glob(luadFilesPaths+"/*.png"))

totalTrainList = []
totalTestList = []



for filename in fileLists:
    imageName = filename.split("/")[-1]
    if len(imageName) > 16:
        totalTestList.append(imageName)
    else:
        totalTrainList.append(imageName)

random.shuffle(totalTrainList)
random.shuffle(totalTestList)

f = open("/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/newSplit/1.training_20230304.txt", "w")
for fl in totalTrainList:
    f.writelines(fl+"\n")
f.close()

f = open("/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/newSplit/1.testing_20230304.txt", "w")
for fl in totalTestList:
    f.writelines(fl+"\n")
f.close()