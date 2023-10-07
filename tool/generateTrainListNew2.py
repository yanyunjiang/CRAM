import os
from glob import glob
import random
from tqdm import tqdm

totalTestList = []
for fold_name in ["19-del+S1403717-", "19-del+S1704712", "L858R+S1406197"]:


    luadFilesPaths = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/newSplit/"+fold_name

    fileLists = sorted(glob(luadFilesPaths+"/*/*.png"))

    # totalTrainList = []

    for filename in tqdm(fileLists):
        imageName = filename.split("/")[-1]
        # if len(imageName) > 16:
        totalTestList.append(fold_name+"/"+filename.split("/")[-2]+"/"+imageName)


# random.shuffle(totalTrainList)
# random.shuffle(totalTestList)

# f = open("/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/newSplit/1.training_20230304.txt", "w")
# for fl in totalTrainList:
#     f.writelines(fl+"\n")
# f.close()

f = open("/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/newSplit/3.testing_shengli_20230312.txt", "w")
for fl in totalTestList:
    f.writelines(fl+"\n")
f.close()