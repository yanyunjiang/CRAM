import os
from glob import glob
import random
from PIL import Image
import numpy as np

# luadFilesPaths = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/3.testing/img"

resultsFilesPaths = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/mask"

fileLists = sorted(glob(resultsFilesPaths+"/*.png"))

Total = []

for filename in fileLists:
    t, s, n = filename[-12:-11], filename[-9:-8], filename[-6:-5]
    # print("t:", t, "s:", s, "n:", n)
    if t == "1" and s == "1":
        img = Image.open(filename)
        arr = np.unique(img)
        if arr[0] == 0 and arr[1] == 1:
            nums = np.sum(np.array(img) == np.ones(np.array(img).shape))
            if nums > 2000:
                Total.append(filename.split("/")[-1])
        else:
            continue 
    else:
        Total.append(filename.split("/")[-1])

random.shuffle(Total)
f = open("/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.trainseg_list2.txt", "w")
for fl in Total:
    f.writelines(fl+"\n")
f.close()