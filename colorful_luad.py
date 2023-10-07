import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm 


def colorful(img_name, img_path, out_path):
    # print(img_name)
    name = os.path.join(img_path, img_name + '.png')
    im = Image.open(name)
    # print("type im", type(im))
    out_path = os.path.join(out_path, img_name + '.png')
    
    # im = np.numpy(im)
    im = np.array(im, dtype = np.uint8)
    # print("2", np.unique(im))

    im0 = im - 1 
    # print("3", np.unique(im0))
    # if im0 == np.ones(im0)*(-1) :
    im = (im0 == (np.ones(im0.shape)*255)) * 3 + im0*(im0 != (np.ones(im0.shape)*255))
    # print(np.unique(im))
    im = Image.fromarray(np.uint8(im))
    # print("type2 im", type(im))
    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 21] = np.array([[0, 64, 128],
                                 [64, 128, 0],
                                 [243, 152, 0],
                                 [255, 255, 2555],
                                 [0, 0, 128],
                                 [128, 0, 128],
                                 [0, 128, 128],
                                 [128, 128, 128],
                                 [64, 0, 0],
                                 [192, 0, 0],
                                 [64, 128, 0],
                                 [192, 128, 0],
                                 [64, 0, 128],
                                 [192, 0, 128],
                                 [64, 128, 128],
                                 [192, 128, 128],
                                 [0, 64, 0],
                                 [128, 64, 0],
                                 [0, 192, 0],
                                 [128, 192, 0],
                                 [0, 64, 128]
                                 ], dtype='uint8').flatten()

    im.putpalette(palette)
    im.save(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default=None, type=str)
    parser.add_argument("--out_path", default=None, type=str)
    args = parser.parse_args()
    img_name_list = os.listdir(args.img_path)
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    for img_name in tqdm(img_name_list):
        if img_name[-4:] != '.png':
            continue
        img_name = img_name[:-4]
        colorful(img_name, args.img_path, args.out_path)