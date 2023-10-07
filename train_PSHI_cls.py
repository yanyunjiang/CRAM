import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils
import argparse
import importlib
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
from prefetch_generator import BackgroundGenerator
from DenseEnergyLoss import DenseEnergyLoss
from tool.myTool import compute_seg_label, compute_joint_loss, compute_cam_up
import datetime

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':
    # starttime = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=1, type=int)
    parser.add_argument("--network", default="network.resnet38_EDAM_cls", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=20, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--train_list", default="voc12/luad_train.txt", type=str)
    parser.add_argument("--session_name", default="Luad", type=str)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--voc12_root", default="/WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training", type=str)
    parser.add_argument("--model_path", default="/WeaklySupervisedSemanticSegmentation/Model/model_luad", type=str)
    parser.add_argument('--densecrfloss', type=float, default=1e-7,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale', type=float, default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb', type=float, default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy', type=float, default=100.0,
                        help='DenseCRF sigma_xy')
    args = parser.parse_args()
    
    writer = SummaryWriter(flush_secs=10)
    model = getattr(importlib.import_module(args.network), 'Net')()

    pyutils.Logger("./logs/" + args.session_name + '.log')

    print(vars(args))

    train_dataset = voc12.data.VOC12ClsDatasetLuad(args.train_list, voc12_root=args.voc12_root,
                        transform=transforms.Compose([
                        imutils.RandomResizeLong(150, 300),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray]),
                    #     model.normalize,
                    #     imutils.RandomCrop(args.crop_size),
                    #     imutils.HWC_to_CHW,
                    #     torch.from_numpy
                    # ]),
                        transform2=
                        imutils.Compose([imutils.Normalize(mean=(0.736, 0.505, 0.678), std=(0.178, 0.221, 0.156)),
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW])
                    )

    train_data_loader = DataLoaderX(train_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': args.lr * 2, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': args.lr * 10, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': args.lr * 20, 'weight_decay': 0},
        {'params': param_groups[4], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[5], 'lr': args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step, warm_up_step=2000)

    if args.weights[-7:] == '.params':
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    critersion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='elementwise_mean').cuda()
    DenseEnergyLosslayer = DenseEnergyLoss(weight=args.densecrfloss, sigma_rgb=args.sigma_rgb,
                                           sigma_xy=args.sigma_xy, scale_factor=args.rloss_scale)
    loss_function = torch.nn.BCEWithLogitsLoss().cuda()
    torch.backends.cudnn.benchmark = True
    
    if not os.path.exists(os.path.join(args.model_path, args.session_name)):
        os.makedirs(os.path.join(args.model_path, args.session_name))

    # s1 = datetime.datetime.now()
    # print ("s1: ", (s1 - starttime))
    for ep in range(args.max_epoches):
        # s2 = datetime.datetime.now()
        for iter, pack in enumerate(train_data_loader):
            # s3 = datetime.datetime.now()
            name = pack[0]

            img = pack[1].cuda(non_blocking=True) # torch.Size([4, 2, 3, 368, 368]) 
            b, a, c, w, h = img.shape
            label = pack[2]   # torch.Size([4, 40])
            # label1 = label.view(-1)  # torch.Size([160])
            label = label.view(-1, 3).cuda(non_blocking=True) # torch.Size([8, 20])
            # print("name:", name,"label:", label)
            label_idx = pack[3]  # len(label_idx) = 20   [tensor([11,  3,  8,  2]), tensor([13,  4, 14, 13]), tensor([14,  8, 17,  8]), tensor([19, 14, 18,  9]), tensor([ 7,  1, 10, 18]), tensor([ 5, 17,  9, 19]), tensor([10, 10, 12, 12]),
            ori_images = pack[4].permute(0, 1, 4, 2, 3).view(-1, c, h, w).numpy() # torch.Size([8, 3, 368, 368])

            croppings = pack[5].view(-1, h, w).numpy().transpose(1,2,0) # torch.Size([368, 368, 8])
            # print("cropping.shape:", croppings.shape)
            
            label_true = pack[6].view(-1, 3).cuda(non_blocking=True)
            # print("label_true.shape: ", label_true.shape)
            

            # x = model(img, label_idx, require_seg=False)
            # # print("x.shape: ", x.shape)
            # closs = loss_function(x, label1)
            # loss = closs
            # s4 = datetime.datetime.now()
            # print ("s2: ", (s4 - s3))
            # if False: #(optimizer.global_step - 1) < 0.5*optimizer.max_step:
            # # if (optimizer.global_step - 1) < 500:
            #     x = model(img, label_idx, require_seg=False)
            #     closs = loss_function(x, label)
            #     loss = closs
            #     # print('closs', closs.data)
            # else:

            x_out, cam, seg = model(img, label_idx, require_seg=True)   
            # x.shape: torch.Size([8, 20])
            # cam: Size([8, 20, 46, 46])
            # seg: Size([8, 21, 46, 46])
            # len(label) = 160   tensor([1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0 ,,,,,])
            # s5 = datetime.datetime.now()
            # print ("s3: ", (s5 - s4))
            cam_up = compute_cam_up(cam, label_true, w, h, b*a) 
            seg_label = np.zeros((b*a, w, h))
            cam_weight = np.zeros((b*a, w, h))
            # s6 = datetime.datetime.now()
            # print ("s4: ", (s6 - s5))
            for i in range(b*a):
                # s6_1 = datetime.datetime.now()
                cam_up_single = cam_up[i]  #  (20, 448, 448)
                cam_label = label_true[i].cpu().numpy()  # (20,)
                ori_img = ori_images[i].transpose(1, 2, 0).astype(np.uint8)  # (448, 448, 3)
                norm_cam = cam_up_single / (np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)  # (20, 448, 448)
                # s6_2 = datetime.datetime.now()
                # print ("i: ",i, "s4_1: ", (s6_2 - s6_1))
                seg_label[i] = compute_seg_label(ori_img, cam_label, norm_cam)
                # s6_3 = datetime.datetime.now()
                # print ("i: ",i, "s4_2: ", (s6_3 - s6_2))

            # s7 = datetime.datetime.now()
            # print ("s5: ", (s7 - s6))
            closs = loss_function(x_out, label)
            # celoss, dloss = compute_joint_loss(ori_images, seg, seg_label, croppings, critersion, DenseEnergyLosslayer)

            bg_celoss, fg_celoss, dloss = compute_joint_loss(ori_images, seg, seg_label, croppings, critersion, DenseEnergyLosslayer)

            # loss = closs + 0.5*celoss + dloss
            loss = closs + 0.5*bg_celoss + 0.5*fg_celoss + dloss
            # loss = closs + bg_celoss + fg_celoss + dloss

            if torch.isnan(bg_celoss):
                print("name:", name)
            
            # print('closs: %.4f'% closs.item(),'celoss: %.4f'%celoss.item(), 'dloss: %.4f'%dloss.item())

            print('closs: %.4f'% closs.item(),'bg_celoss: %.4f'%bg_celoss.item(),'fg_celoss: %.4f'%fg_celoss.item(), 'dloss: %.4f'%dloss.item())

            # s8 = datetime.datetime.now()
            # print ("s6: ", (s8 - s7))
            avg_meter.add({'loss': loss.item()})
            writer.add_scalar('Train/Loss', loss.item(), optimizer.global_step)
            writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], optimizer.global_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1)%50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                    'Loss:%.4f' % (avg_meter.pop('loss')),
                    'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                    'Fin:%s' % (timer.str_est_finish()),
                    'lr: %.6f' % (optimizer.param_groups[0]['lr']), flush=True)
            if optimizer.global_step % (max_step//25) == 0:
                ep = int(optimizer.global_step // 1322)
                torch.save(model.module.state_dict(), os.path.join(args.model_path, args.session_name, str(ep) + '.pth'))
            
        # s9 = datetime.datetime.now()
        # print ("s7: ", (s9 - s2))

        # t1 = datetime.datetime.now()
    
    torch.save(model.module.state_dict(), os.path.join(args.model_path, args.session_name, 'final.pth'))
    writer.close()

