import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import imageio
import importlib
from tool import imutils
import argparse
import cv2
import os.path
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from tqdm import tqdm


def _crf_with_alpha(pred_prob, ori_img):
    bgcam_score = pred_prob.cpu().data.numpy()
    crf_score = imutils.crf_inference_inf(ori_img, bgcam_score, labels=3)

    return crf_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='./netWeights/RRM_final.pth', type=str)
    parser.add_argument("--network", default="network.resnet38_EDAM_cls", type=str)
    parser.add_argument("--out_cam_pred", default='../../Model/model_edam/seg_out/no_crf_edam', type=str)
    parser.add_argument("--out_la_crf", default='../../Model/model_edam/seg_out/crf_edam', type=str)
    parser.add_argument("--LISTpath", default="./voc12/luad_train_list.txt", type=str)
    parser.add_argument("--IMpath", default="/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/img", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.out_cam_pred):
        os.makedirs(args.out_cam_pred)
    if not os.path.exists(args.out_la_crf):
        os.makedirs(args.out_la_crf)

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))
    

    model.eval()
    model.cuda()
    im_path = args.IMpath
    img_list = open(args.LISTpath).readlines()
    pred_softmax = torch.nn.Softmax(dim=0)
    for i in tqdm(img_list):
        # print("i:", i)
        image_name = os.path.join(im_path, i[:-1])
        name = image_name.split("/")[-1]
        # print("name:", name)
        if name != "436219-35356-55497-[1, 1, 0].png":
            continue
        else:
            print("name:", name)
            img_temp = cv2.imread(image_name)
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)
            img_original = img_temp.astype(np.uint8)
            img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
            img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
            img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

            input = torch.from_numpy(img_temp[np.newaxis, :].transpose(0, 3, 1, 2)).float().cuda()
            # print("input.shape:", input.shape)
            # output = model.forward_seg(input)
            cam, cam2, map2 = model.forward_cam(input)
            output = F.interpolate(map2, (img_temp.shape[0], img_temp.shape[1]),mode='bilinear',align_corners=False)
            # output = output.permute(0,1,2,3)
            image_cam = output.detach().cpu().numpy().astype(np.uint8)
            print("image_cam.shape:", image_cam[0][0])
            print("image_cam.shape[0]:", image_cam.shape[0])
            for i in range(image_cam.shape[1]):
                save_path = os.path.join(args.out_cam_pred,"{}.png".format(str(i)))
                cv2.imwrite(save_path, image_cam[0][i])




            quit()
            output = torch.squeeze(output)
            pred_prob = pred_softmax(output)
            output = torch.argmax(output,dim=0).cpu().numpy()

            save_path = os.path.join(args.out_cam_pred,i[:-1])
            cv2.imwrite(save_path,output.astype(np.uint8))

            if args.out_la_crf is not None:
                crf_la = _crf_with_alpha(pred_prob, img_original)

                crf_img = np.argmax(crf_la, 0)

                imageio.imsave(os.path.join(args.out_la_crf, i[:-1]), crf_img.astype(np.uint8))

