import torch
import torch.nn as nn
import torch.nn.functional as F
from network.transformer import TransformerBlock
from network.embedding import BERTEmbedding
import network.resnet38d

import cv2, os
import numpy as np
from PIL import Image


class Net(network.resnet38d.Net):
    def __init__(self):
        super().__init__()
        self.mask_layer = nn.Conv2d(4096, 3, 3, padding=1, bias=False)
        torch.nn.init.xavier_uniform_(self.mask_layer.weight)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout7 = torch.nn.Dropout2d(0.8)

        self.sample_num = 3

        self.hidden = 128
        self.mini_batch_size = 2
        self.attn_heads = 2
        self.n_layers = 1
        
        self.embedding = BERTEmbedding(embed_size=self.hidden, mini_batch_size=self.mini_batch_size, sample_num=self.sample_num)
        self.trans_list = nn.ModuleList([nn.ModuleList(
            [TransformerBlock(self.hidden, self.attn_heads, self.hidden * 4, 0.1) for _ in range(self.n_layers)]) for _ in range(20)])

        self.d_d = nn.Conv2d(4096, self.hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.d_d.weight)

        ###### seg networks ######
        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)

        self.fc8_seg_conv2 = nn.Conv2d(512, 21, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)
        ##########################

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.mask_layer, self.d_d, self.fc8_seg_conv1, self.fc8_seg_conv2]

        self.fc_list = nn.ModuleList([nn.Conv2d(self.hidden, 1, 1, bias=False) for _ in range(20)])

        for fc in self.fc_list:
            torch.nn.init.xavier_uniform_(fc.weight)
            self.from_scratch_layers.append(fc)

        for transformer in self.trans_list:
            for p in transformer.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)

        for p in self.embedding.parameters():
            torch.nn.init.xavier_uniform_(p)



    def forward(self, img, label_idx=None, require_seg = False):
        # label_idx 长度20，
        c, h, w = img.size()[-3:]   # img.shape: torch.Size([1, 2, 3, 368, 368])
        x = img.view(-1, c, h, w)  # x.shape: torch.Size([2, 3, 368, 368])
        xo = super().forward(x)  # x.shape: torch.Size([2, 4096, 46, 46])
        # x_cam = xo.clone()
        # x_seg = xo.clone()

        


        n, c, h, w = xo.size()
        assert n == self.mini_batch_size
        assert len(label_idx) == self.sample_num

        ######## cam 操作模块1 ########
        # x_cam = xo.clone()
        mask = self.mask_layer(xo)  # mask.shape: torch.Size([2, 20, 46, 46])
        # cam = F.relu(mask)
        mask = F.normalize(mask)  # mask.shape: torch.Size([2, 20, 46, 46])
        cam = F.relu(mask)
        mask = torch.abs(mask)  # 也是cam输出
        # cam = mask

        

        # print("mask.shape:", mask.shape)
        # cam = mask
        # x_cam = xo.clone()
        # mask = self.mask_layer(x_cam)

        #############################

        label_feature = []
        x = self.d_d(xo) # x.shape: torch.Size([2, 128, 46, 46])
        # xc1 = F.relu(x)
        # xc2 = self.d_d_cam(xc1)
        # cam = F.relu(xc2)

        ######## cam 操作模块2 ########
        # x_cam = xo.clone()
        # cam = F.conv2d(x_cam, self.d_d.weight)
        # cam = F.relu(cam)
        # print("cam.shape:", cam.shape)
        #############################

        x = x.permute(1, 0, 2, 3).contiguous()         # self.hidden,n,h,w ==> torch.Size([128, 2, 46, 46])
        mask = mask.permute(1, 0, 2, 3).contiguous()        # 21,n,h,w ==> torch.Size([21, 2, 46, 46])

        for i in label_idx:
            # x.shape:  torch.Size([128, 2, 46, 46]),   mask[i].shape:  torch.Size([1, 2, 46, 46])
            feature = torch.mul(x, mask[i]).permute(1, 0, 2, 3) # n,c,h,w  torch.Size([128, 2, 46, 46]) ==> torch.Size([2, 128, 46, 46])
            # print("feature.shape:", feature.shape)
            label_feature.append(feature)
        x = torch.stack(label_feature, 1).view(-1, self.hidden, h, w)   # n*self.sample_num,self.hidden,h,w ==> torch.Size([40, 128, 46, 46])

        x = x.view(n, self.sample_num, self.hidden, h, w).permute(0,1,3,4,2).contiguous()  # torch.Size([2, 20, 46, 46, 128])
        x = x.view(1, self.mini_batch_size*self.sample_num*h*w, self.hidden) # torch.Size([1, 84640, 128])

        segment_info = torch.zeros(1, self.mini_batch_size * self.sample_num * h * w) # torch.Size([1, 84640])
        for i in range(self.mini_batch_size):
            segment_info[:, i * self.sample_num * h * w:(i + 1) * self.sample_num * h * w] = i
        segment_info = segment_info.to(torch.int64).cuda() # torch.Size([1, 84640])
        
        x = self.embedding(x, segment_info) # torch.Size([1, 84640, 128])
        
        output = []
        for i in range(self.sample_num):
            trans_input = []
            for j in range(self.mini_batch_size):
                trans_input.append(x[:, i*h*w+j*self.sample_num*h*w:(i+1)*h*w+j*self.sample_num*h*w, :]) 
            trans_input = torch.stack(trans_input, dim=1).view(1, -1, self.hidden)     # 1,self.mini_batch*h*w,self.hidden == ([1, 4232, 128])
            for block in self.trans_list[label_idx[i]]:
                trans_input = block.forward(trans_input, None)   # torch.Size([1, 4232, 128])
                
            # trans_input.shape: torch.Size([1, 4232, 128])
            trans_output = trans_input.view(self.mini_batch_size, h, w, self.hidden).permute(0,3,1,2) # shape: torch.Size([2, 128, 46, 46])
            trans_output = self.gap(trans_output)  # torch.Size([2, 128, 1, 1])
            trans_output = self.fc_list[label_idx[i]](trans_output).view(-1) # torch.Size([2])
            output.append(trans_output) 
        
        # len(output) ==> 20    output: [tensor([-1.4948, -5.1446], device='cuda:0', grad_fn=<ViewBackward0>), tensor([-5.1989, -2.6877], device='cuda:0', grad_fn=<ViewBackward0>), tensor([18.7119, 14.2116], device='cuda:0', grad_fn=<ViewBackward0>), ... ]

        ########################
        # x = torch.stack(output, dim=1).view(-1)  # self.mini_batch_size*self.sample_num ==> torch.Size([40]) x_out: tensor([ -1.4948,  -5.1989,  18.7119,  -9.1775,  -7.9660,  -4.0605,  -9.6484, ****])

        x = torch.stack(output, dim=1)

        if require_seg:
            # 添加 分割layer
            #############################
            x_seg = xo.clone()
            x_seg = self.fc8_seg_conv1(x_seg)
            x_seg = F.relu(x_seg)
            seg = self.fc8_seg_conv2(x_seg)
            #############################

            return x, cam, seg
        else:
            return x

    def forward_cam(self, x):
        x = super().forward(x)

        ## 画x0伪彩色图
        # xo = x
        # output_ = F.interpolate(xo, (460, 460),mode='bilinear',align_corners=False)  # x.shape: torch.Size([2, 4096, 460, 460])
        # print("output_.shape: ", output_.shape)
        # for i in range(500):
        #     im_color = cv2.applyColorMap((output_[0][i]*100).detach().cpu().numpy().astype(np.uint8), 2)
        #     new_img = Image.fromarray(im_color)
        #     new_img.save(os.path.join("/WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226_5featuremap", "{}.png".format(str(i))))
        # quit(0)

        mask0 = self.mask_layer(x)
        mask = F.normalize(mask0)
        # cam = F.relu(mask)
        
        # mask = F.normalize(mask)
        cam = torch.abs(mask)

        # x = F.conv2d(x, self.fc8.weight)
        # cam = F.relu(x)

        # x = self.d_d(x) # x.shape: torch.Size([2, 128, 46, 46])
        # xc1 = F.relu(x)
        # xc2 = self.d_d_cam(xc1)
        # cam = F.relu(xc2)

        cam2 = F.relu(mask0)

        return cam  #, cam2, x

    def forward_seg(self, x):
        x_seg = super().forward(x)
        x_seg = self.fc8_seg_conv1(x_seg)
        x_seg = F.relu(x_seg)
        seg = self.fc8_seg_conv2(x_seg)
        return seg

    def get_parameter_groups(self):
        groups = ([], [], [], [], [], [])
        for name,m in self.embedding.named_parameters():
            if m.requires_grad:
                groups[4].append(m)
        for transformer in self.trans_list:
            for name, m in transformer.named_parameters():
                if m.requires_grad:
                    groups[4].append(m)
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        for i in range(len(groups)):
            print(len(groups[i]))

        return groups
