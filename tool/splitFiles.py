import os
from glob import glob
import random

luadFilesPaths = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training"

fileLists = glob(luadFilesPaths+"/*.png")

Total = []
Tumor = []
Stroma = []
Normal = []

choseNums = 12

for filename in fileLists:
    filen = filename.split('/')[-1]
    label = filename.split('/')[-1][-13:-4]
    t = label[1]
    s = label[4]
    n = label[7]
    
    if t == "1":
        Tumor.append(filen)
    if s == "1":
        Stroma.append(filen)
    if n == "1":
        Normal.append(filen)
# print("len Tumor:", len(Tumor))
# TumorCopy = Tumor
for i in range(len(Tumor)):
    TumorCopy = Tumor.copy()
    a = Tumor[i]
    del TumorCopy[i]
    b = random.sample(TumorCopy, choseNums)
    for j in range(0, choseNums):
        if j < choseNums/2:
            Total.append([a, b[j], 0])
        else:
            Total.append([b[j], a, 0])

for i in range(len(Stroma)):
    StromaCopy = Stroma.copy()
    a = Stroma[i]
    del StromaCopy[i]
    b = random.sample(StromaCopy, choseNums)
    for j in range(0, choseNums):
        if j < choseNums/2:
            Total.append([a, b[j], 1])
        else:
            Total.append([b[j], a, 1])

for i in range(len(Normal)):
    NormalCopy = Normal.copy()
    a = Normal[i]
    del NormalCopy[i]
    b = random.sample(NormalCopy, choseNums)
    for j in range(0, choseNums):
        if j < choseNums/2:
            Total.append([a, b[j], 2])
        else:
            Total.append([b[j], a, 2])


random.shuffle(Total)
f = open("luad_train.txt", "w")
for fl in Total:
    f.writelines(fl[0]+";"+fl[1]+";"+str(fl[2])+"\n")
f.close()


