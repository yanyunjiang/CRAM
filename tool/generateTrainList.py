import os
from glob import glob
import random

luadFilesPaths = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/2.validation/img_split"

fileLists = sorted(glob(luadFilesPaths+"/*.png"))

Total = []

for filename in fileLists:
    Total.append(filename.split("/")[-1])

random.shuffle(Total)
f = open("/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/2.validation_20230217.txt", "w")
for fl in Total:
    f.writelines(fl+"\n")
f.close()