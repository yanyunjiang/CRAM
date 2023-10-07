import cv2
import numpy as np 
from PIL import Image

# im = Image.open("image.png") # Replace with your image name here
# indexed = np.array(im) # Convert to NumPy array to easier access


image_path = "/WeaklySupervisedSemanticSegmentation/Code/DeepLabV3Plus-Pytorch/test_results/best_deeplabv3plus_resnet50_luad_os16_sxd1_Scale2.pth/00.png"



image_path1 = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/2.validation/img/01.png"

image_path1 = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/2.validation/mask_index/02.png"

image_path1 = "/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/1.training_index/436219-33187-43407-[1, 0, 0].png"

# [img, cmap] = cv2.imread(image_path1)
im = Image.open(image_path1).convert('L')
im = np.array(im)
print(im.shape)
print(np.unique(im))