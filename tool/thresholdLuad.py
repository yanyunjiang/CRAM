import os
from glob import glob
from tqdm import tqdm
import cv2

luadFilesPath = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training"
outThresFilePath = "/WeaklySupervisedSemanticSegmentation/Model/model_luad/thresHoldImagesBinary"


luadFileList = glob(luadFilesPath+"/*.png")

for fileName in tqdm(luadFileList):

    img = cv2.imread(fileName)
    # print(fileName)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    min_value=215
    #固定阈值
    ret, thresh1 = cv2.threshold(gray, min_value, 255, cv2.THRESH_BINARY)
    # color = cv2.cvtColor(thresh1,cv2.GRAY2COLOR_BGR)
    # print(color.shape)
    cv2.imwrite(outThresFilePath+"/"+fileName.split("/")[-1], 255-thresh1)
    
    
