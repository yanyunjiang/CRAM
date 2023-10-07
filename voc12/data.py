import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import random
import pyarrow as pa


IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))


def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/luad_labels.npy', allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]


def load_image_label_pair_list_from_npy(img_name_pair_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

    return [(cls_labels_dict[img_name_pair[0]], cls_labels_dict[img_name_pair[1]]) for img_name_pair in img_name_pair_list]

def load_image_label_pair_list_from_npy_luad(img_name_pair_list):

    cls_labels_dict = np.load('voc12/luad_labels.npy', allow_pickle=True).item()

    return [(cls_labels_dict[img_name_pair[0]], cls_labels_dict[img_name_pair[1]]) for img_name_pair in img_name_pair_list]


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, img_name)
    # return os.path.join(voc12_root, img_name + '.jpg')

def get_img_path_luad(img_name, voc12_root):
    return os.path.join(voc12_root, img_name)

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name for img_gt_name in img_gt_name_list]
    # print("img_name_list", img_name_list)
    return img_name_list

# def load_img_name_list_luad(dataset_path):

#     img_gt_name_list = open(dataset_path).read().splitlines()
#     img_name_list = [img_gt_name for img_gt_name in img_gt_name_list]

#     return img_name_list

def load_img_name_pair_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    # img_name_pair_list = [(img_gt_name.split(' ')[0][-15:-4], img_gt_name.split(' ')[1][-15:-4]) for img_gt_name in img_gt_name_list]
    # common_label_list = [int(img_gt_name.split(' ')[2]) for img_gt_name in img_gt_name_list]
    img_name_pair_list = [(img_gt_name.split(' ')[0][-15:-4], img_gt_name.split(' ')[1][-15:-4]) for img_gt_name in
                          img_gt_name_list]
    common_label_list = [int(img_gt_name.split(' ')[2]) for img_gt_name in img_gt_name_list]

    return img_name_pair_list, common_label_list

def load_img_name_pair_list_luad(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    # img_name_pair_list = [(img_gt_name.split(' ')[0][-15:-4], img_gt_name.split(' ')[1][-15:-4]) for img_gt_name in img_gt_name_list]
    # common_label_list = [int(img_gt_name.split(' ')[2]) for img_gt_name in img_gt_name_list]
    img_name_pair_list = [(img_gt_name.split(';')[0], img_gt_name.split(';')[1]) for img_gt_name in img_gt_name_list]
    common_label_list = [int(img_gt_name.split(';')[2]) for img_gt_name in img_gt_name_list]

    return img_name_pair_list, common_label_list


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label


class VOC12EDAMClsDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_pair_list, self.common_label_list = load_img_name_pair_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.label_pair_list = load_image_label_pair_list_from_npy(self.img_name_pair_list)
        # print("self.label_pair_list", self.label_pair_list)

    def __len__(self):
        return len(self.img_name_pair_list)

    def __getitem__(self, idx):
        name_pair = self.img_name_pair_list[idx]
        # print("name_pair:", name_pair)
        img1 = PIL.Image.open(get_img_path(name_pair[0], self.voc12_root)).convert("RGB")
        img2 = PIL.Image.open(get_img_path(name_pair[1], self.voc12_root)).convert("RGB")

        if self.transform:
            img_pair = torch.stack((self.transform(img1), self.transform(img2)), dim=0)

        label1 = torch.from_numpy(self.label_pair_list[idx][0]) # img1图像对应的标签，一幅图像有多个标签
        label2 = torch.from_numpy(self.label_pair_list[idx][1]) # img2图像对应的标签，一幅图像有多个标签
        # print("label1: ", label1)
        # print("label2: ", label2)

        

        commen_label = self.common_label_list[idx] # 一对图像的公共label标签

        list1 = [i for i in range(20) if label1[i] == 0 and label2[i] == 0]
        print("list1: ", list1)
        random.shuffle(list1)
        print("list1_new: ", list1)
        list2 = [i for i in range(20) if label1[i] == 1 or label2[i] == 1]
        print("list2: ", list2)
        label_idx = list2
        print("label_idx: ", label_idx)
        sample_num = 20
        if len(label_idx) > sample_num:
            label_idx = label_idx[:sample_num]

        for i in range(0, -len(label_idx)+sample_num):
            print("i", i, "list1[i]", list1[i])
            label_idx.append(list1[i])

        assert label1[commen_label] == 1 and label2[commen_label] == 1
        assert len(label_idx) == len(set(label_idx))
        list3 = [0 for _ in range(2*sample_num)]
        print("list3: ", len(list3))
        print("label_idx**: ", label_idx)
        for i in range(sample_num):
            if label1[label_idx[i]] == 1:
                list3[i] = 1.0
            if label2[label_idx[i]] == 1:
                list3[i+sample_num] = 1.0
        label = torch.tensor(list3)
        print("label x: ", label)
        print("label len(x): ", len(label))

        return name_pair, img_pair, label, label_idx

class VOC12ClsDatasetEnd2end(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None, transform2=None):
        self.img_name_pair_list, self.common_label_list = load_img_name_pair_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.transform2 = transform2
        self.label_pair_list = load_image_label_pair_list_from_npy(self.img_name_pair_list)

    def __len__(self):
        return len(self.img_name_pair_list)

    def __getitem__(self, idx):
        name_pair = self.img_name_pair_list[idx]
        # print("name_pair:", name_pair)
        img1 = PIL.Image.open(get_img_path(name_pair[0], self.voc12_root)).convert("RGB")
        img2 = PIL.Image.open(get_img_path(name_pair[1], self.voc12_root)).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            # img_pair = torch.stack((img1, img2), dim=0)

        if self.transform2:
            ori_img1 = img1
            ori_img2 = img2
            croppings1 = np.zeros_like(img1)
            croppings2 = np.zeros_like(img2)
            img_dict1 = []
            img_dict2 = []
            img_dict1.append(img1)
            img_dict2.append(img2)
            img_dict1.append(ori_img1)
            img_dict2.append(ori_img2)
            img_dict1.append(croppings1)
            img_dict2.append(croppings2)
            img1, ori_img1, croppings1 = self.transform2(img_dict1)
            img2, ori_img2, croppings2 = self.transform2(img_dict2)

            img_pair = torch.stack(( torch.from_numpy(img1),  torch.from_numpy(img2)), dim=0)
            ori_img_pair = torch.stack(( torch.from_numpy(ori_img1),  torch.from_numpy(ori_img2)), dim=0)
            croppings_pair = torch.stack(( torch.from_numpy(croppings1),  torch.from_numpy(croppings2)), dim=0)


        label1 = torch.from_numpy(self.label_pair_list[idx][0]) # img1图像对应的标签，一幅图像有多个标签
        label2 = torch.from_numpy(self.label_pair_list[idx][1]) # img2图像对应的标签，一幅图像有多个标签
        # print("name_pair[0]", name_pair[0], "label1: ", label1)
        # print("name_pair[1]", name_pair[1], "label2: ", label2)

        label_true = torch.stack((label1, label2), dim=0)
        # print("label_true.shape:", label_true.shape)

        commen_label = self.common_label_list[idx] # 一对图像的公共label标签

        list1 = [i for i in range(20) if label1[i] == 0 and label2[i] == 0]
        # print("list1: ", list1)
        random.shuffle(list1)
        # print("list1_new: ", list1)
        list2 = [i for i in range(20) if label1[i] == 1 or label2[i] == 1] # 输出label的标签位置
        # print("list2: ", list2)
        label_idx = list2
        # print("label_idx: ", label_idx)
        sample_num = 20
        if len(label_idx) > sample_num:

            label_idx = label_idx[:sample_num]

        for i in range(0, -len(label_idx)+sample_num):
            # print("i", i, "list1[i]", list1[i])
            label_idx.append(list1[i])

        assert label1[commen_label] == 1 and label2[commen_label] == 1
        assert len(label_idx) == len(set(label_idx))
        list3 = [0 for _ in range(2*sample_num)]
        # print("list3: ", len(list3))
        # print("label_idx**: ", label_idx)
        for i in range(sample_num):
            if label1[label_idx[i]] == 1:
                list3[i] = 1.0
            if label2[label_idx[i]] == 1:
                list3[i+sample_num] = 1.0
        label = torch.tensor(list3)
        # print("label x: ", label)
        # print("label len(x): ", len(label))
        # quit(1)
        return name_pair, img_pair, label, label_idx, ori_img_pair, croppings_pair, label_true


class VOC12ClsDatasetLuad(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None, transform2=None):
        self.img_name_pair_list, self.common_label_list = load_img_name_pair_list_luad(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.transform2 = transform2
        self.label_pair_list = load_image_label_pair_list_from_npy_luad(self.img_name_pair_list)
        # print("self.label_pair_list", self.label_pair_list)

    def __len__(self):
        return len(self.img_name_pair_list)

    def __getitem__(self, idx):
        name_pair = self.img_name_pair_list[idx]
        # print("name_pair:", name_pair)
        img1 = PIL.Image.open(get_img_path_luad(name_pair[0], self.voc12_root)).convert("RGB")
        img2 = PIL.Image.open(get_img_path_luad(name_pair[1], self.voc12_root)).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            # img_pair = torch.stack((img1, img2), dim=0)

        if self.transform2:
            ori_img1 = img1
            ori_img2 = img2
            croppings1 = np.zeros_like(img1)
            croppings2 = np.zeros_like(img2)
            img_dict1 = []
            img_dict2 = []
            img_dict1.append(img1)
            img_dict2.append(img2)
            img_dict1.append(ori_img1)
            img_dict2.append(ori_img2)
            img_dict1.append(croppings1)
            img_dict2.append(croppings2)
            img1, ori_img1, croppings1 = self.transform2(img_dict1)
            img2, ori_img2, croppings2 = self.transform2(img_dict2)

            img_pair = torch.stack(( torch.from_numpy(img1),  torch.from_numpy(img2)), dim=0)
            ori_img_pair = torch.stack(( torch.from_numpy(ori_img1),  torch.from_numpy(ori_img2)), dim=0)
            croppings_pair = torch.stack(( torch.from_numpy(croppings1),  torch.from_numpy(croppings2)), dim=0)


        label1 = torch.from_numpy(self.label_pair_list[idx][0]) # img1图像对应的标签，一幅图像有多个标签
        label2 = torch.from_numpy(self.label_pair_list[idx][1]) # img2图像对应的标签，一幅图像有多个标签
        # print("name_pair[0]", name_pair[0], "label1: ", label1)
        # print("name_pair[1]", name_pair[1], "label2: ", label2)

        label_true = torch.stack((label1, label2), dim=0)
        # print("label_true.shape:", label_true.shape)

        commen_label = self.common_label_list[idx] # 一对图像的公共label标签
        # print("commen_label:", commen_label)
        sample_num = 3

        list1 = [i for i in range(sample_num) if label1[i] == 0 and label2[i] == 0]
        # print("list1: ", list1)
        random.shuffle(list1)
        # print("list1_new: ", list1)
        list2 = [i for i in range(sample_num) if label1[i] == 1 or label2[i] == 1] # 输出label的标签位置
        # print("list2: ", list2)
        label_idx = list2
        # print("label_idx: ", label_idx)
        
        if len(label_idx) > sample_num:
            label_idx = label_idx[:sample_num]

        for i in range(0, -len(label_idx)+sample_num):
            # print("i", i, "list1[i]", list1[i])
            label_idx.append(list1[i])

        # print("label1[commen_label]", label1[commen_label])
        # print("label2[commen_label]", label2[commen_label])
        assert label1[commen_label] == 1 and label2[commen_label] == 1
        assert len(label_idx) == len(set(label_idx))
        list3 = [0 for _ in range(2*sample_num)]
        # print("list3: ", len(list3))
        # print("label_idx**: ", label_idx)
        for i in range(sample_num):
            if label1[label_idx[i]] == 1:
                list3[i] = 1.0
            if label2[label_idx[i]] == 1:
                list3[i+sample_num] = 1.0
        label = torch.tensor(list3)
        # print("label x: ", label)
        # print("label len(x): ", len(label))
        # quit(1)
        return name_pair, img_pair, label, label_idx, ori_img_pair, croppings_pair, label_true



class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

class VOC12ClsDatasetLUADVal(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label


