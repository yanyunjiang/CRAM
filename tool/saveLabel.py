import os
from glob import glob
import random
import numpy as np

# luadFilesPaths = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training"

# fileLists = glob(luadFilesPaths+"/*.png")

# luad_labels = {}

# choseNums = 12

# for filename in fileLists:
#     filen = filename.split('/')[-1]
#     label = filename.split('/')[-1][-13:-4]
#     t = label[1]
#     s = label[4]
#     n = label[7]

#     xx = {filen:np.array([t,s,n], dtype=np.float32)}
#     luad_labels.update(xx)

# np.save("luad_lables.npy", luad_labels)
    

print(np.load("luad_labels.npy", allow_pickle=True))