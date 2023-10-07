
import PIL.Image
import random
import numpy as np
from torchvision import transforms

class Compose(transforms.Compose):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_dict):
        img = img_dict[0]
        ori_img = img_dict[1]
        croppings = img_dict[2]
        for t in self.transforms:
            img, ori_img, croppings = t(img, ori_img, croppings)
        return img, ori_img, croppings

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img, ori_img, croppings):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]
        croppings = np.ones_like(imgarr)
        return proc_img, imgarr, croppings

# class Normalize():
#     def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

#         self.mean = mean
#         self.std = std

#     def __call__(self, img1, img2, ori_img1, ori_img2, croppings1, croppings2):
#         imgarr1 = np.asarray(img1)
#         imgarr2 = np.asarray(img2)
#         proc_img1 = np.empty_like(imgarr1, np.float32)
#         proc_img2 = np.empty_like(imgarr2, np.float32)

#         proc_img1[..., 0] = (imgarr1[..., 0] / 255. - self.mean[0]) / self.std[0]
#         proc_img1[..., 1] = (imgarr1[..., 1] / 255. - self.mean[1]) / self.std[1]
#         proc_img1[..., 2] = (imgarr1[..., 2] / 255. - self.mean[2]) / self.std[2]
#         croppings1 = np.ones_like(imgarr1)

#         proc_img2[..., 0] = (imgarr2[..., 0] / 255. - self.mean[0]) / self.std[0]
#         proc_img2[..., 1] = (imgarr2[..., 1] / 255. - self.mean[1]) / self.std[1]
#         proc_img2[..., 2] = (imgarr2[..., 2] / 255. - self.mean[2]) / self.std[2]
#         croppings2 = np.ones_like(imgarr2)

#         return proc_img1, proc_img2, imgarr1, imgarr2,croppings1, croppings2



class RandomResizeLong():

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img):

        target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        img = img.resize(target_shape, resample=PIL.Image.CUBIC)

        return img

class RandomCrop():

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr, ori_img, croppings):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        ori_contrainer = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        croppings = np.zeros((self.cropsize, self.cropsize), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]
        ori_contrainer[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            ori_img[img_top:img_top+ch, img_left:img_left+cw]
        croppings[cont_top:cont_top+ch, cont_left:cont_left+cw] = 1

        return container, ori_contrainer, croppings

# class RandomCrop():
#     def __init__(self, cropsize):
#         self.cropsize = cropsize

#     def __call__(self, imgarr1, imgarr2, ori_img1, ori_img2, croppings1, croppings2):

#         h, w, c = imgarr1.shape
#         # print("h, w, c", h, w, c)
#         # h1, w1, c1 = imgarr2.shape
#         # print("h1, w1, c1", h1, w1, c1)

#         ch = min(self.cropsize, h)
#         cw = min(self.cropsize, w)

#         w_space = w - self.cropsize
#         h_space = h - self.cropsize

#         if w_space > 0:
#             cont_left = 0
#             img_left = random.randrange(w_space+1)
#         else:
#             cont_left = random.randrange(-w_space+1)
#             img_left = 0

#         if h_space > 0:
#             cont_top = 0
#             img_top = random.randrange(h_space+1)
#         else:
#             cont_top = random.randrange(-h_space+1)
#             img_top = 0

#         container1 = np.zeros((self.cropsize, self.cropsize, imgarr1.shape[-1]), np.float32)
#         container2 = np.zeros((self.cropsize, self.cropsize, imgarr2.shape[-1]), np.float32)

#         ori_contrainer1 = np.zeros((self.cropsize, self.cropsize, imgarr1.shape[-1]), np.float32)
#         ori_contrainer2 = np.zeros((self.cropsize, self.cropsize, imgarr2.shape[-1]), np.float32)

#         croppings1 = np.zeros((self.cropsize, self.cropsize), np.float32)
#         croppings2 = np.zeros((self.cropsize, self.cropsize), np.float32)

#         container1[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
#             imgarr1[img_top:img_top+ch, img_left:img_left+cw]
#         container2[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
#             imgarr2[img_top:img_top+ch, img_left:img_left+cw]
        
#         ori_contrainer1[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
#             ori_img1[img_top:img_top+ch, img_left:img_left+cw]
#         ori_contrainer2[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
#             ori_img2[img_top:img_top+ch, img_left:img_left+cw]

#         croppings1[cont_top:cont_top+ch, cont_left:cont_left+cw] = 1
#         croppings2[cont_top:cont_top+ch, cont_left:cont_left+cw] = 1

#         return container1, container2, ori_contrainer1, ori_contrainer2, croppings1, croppings2

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def crop_with_box(img, box):
    if len(img.shape) == 3:
        img_cont = np.zeros((max(box[1]-box[0], box[4]-box[5]), max(box[3]-box[2], box[7]-box[6]), img.shape[-1]), dtype=img.dtype)
    else:
        img_cont = np.zeros((max(box[1] - box[0], box[4] - box[5]), max(box[3] - box[2], box[7] - box[6])), dtype=img.dtype)
    img_cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
    return img_cont


def random_crop(images, cropsize, fills):
    if isinstance(images[0], PIL.Image.Image):
        imgsize = images[0].size[::-1]
    else:
        imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, fills):

        if isinstance(img, PIL.Image.Image):
            img = img.crop((box[6], box[4], box[7], box[5]))
            cont = PIL.Image.new(img.mode, (cropsize, cropsize))
            cont.paste(img, (box[2], box[0]))
            new_images.append(cont)

        else:
            if len(img.shape) == 3:
                cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
            else:
                cont = np.ones((cropsize, cropsize), img.dtype)*f
            cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
            new_images.append(cont)

    return new_images


class AvgPool2d():

    def __init__(self, ksize):
        self.ksize = ksize

    def __call__(self, img):
        import skimage.measure

        return skimage.measure.block_reduce(img, (self.ksize, self.ksize, 1), np.mean)


class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, img):
        if bool(random.getrandbits(1)):
            img = np.fliplr(img).copy()
        return img


class CenterCrop():

    def __init__(self, cropsize, default_value=0):
        self.cropsize = cropsize
        self.default_value = default_value

    def __call__(self, npimg):

        h, w = npimg.shape[:2]

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        sh = h - self.cropsize
        sw = w - self.cropsize

        if sw > 0:
            cont_left = 0
            img_left = int(round(sw / 2))
        else:
            cont_left = int(round(-sw / 2))
            img_left = 0

        if sh > 0:
            cont_top = 0
            img_top = int(round(sh / 2))
        else:
            cont_top = int(round(-sh / 2))
            img_top = 0

        if len(npimg.shape) == 2:
            container = np.ones((self.cropsize, self.cropsize), npimg.dtype)*self.default_value
        else:
            container = np.ones((self.cropsize, self.cropsize, npimg.shape[2]), npimg.dtype)*self.default_value

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            npimg[img_top:img_top+ch, img_left:img_left+cw]

        return container


# def HWC_to_CHW(img1, img2, ori_img1, ori_img2, croppings1, croppings2):
#     return np.transpose(img1, (2, 0, 1)), np.transpose(img2, (2, 0, 1)), ori_img1, ori_img2, croppings1, croppings2

def HWC_to_CHW_Origin(img):
    return np.transpose(img, (2, 0, 1))

def HWC_to_CHW(img, ori_img, croppings):
    return np.transpose(img, (2, 0, 1)), ori_img, croppings

class RescaleNearest():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, npimg):
        import cv2
        return cv2.resize(npimg, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)




def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def crf_inference_inf(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=83/scale_factor, srgb=5, rgbim=np.copy(img_c), compat=4)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))