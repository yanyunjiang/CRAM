# CRAM
# WSSS-CRAM: Precise Segmentation of Histopathological Images via Class Region Activation Mapping

The inspiration for this code comes from Tong Wu's article.
https://github.com/allenwu97/EDAM EDAM(Embedded Discriminative Attention Mechanism for Weakly Supervised Semantic Segmentation)

## Env
We train our model with Python 3.5, PyTorch 1.1.0 and 4 Tesla A100 GPUs with 40 GB memory.
 
## Dataset
  * The datasets of our experiment were got from [WSSS4LUAD](https://wsss4luad.grand-challenge.org/) and Shandong Provincial Hospital Affiliated to Shandong First Medical University.

## Train
Training PSHI from scratch.
```
python3 train_PSHI_cls.py --lr 0.001 --batch_size 4 --max_epoches 1 --network network.resnet38_PSHI_cls --weights /WeaklySupervisedSemanticSegmentation/Model/pretrain_model/download.params --wt_dec 5e-4 --session_name resnet38_luad_20221206_cam1step0.5ce

```

To monitor loss and lr, run the following command in a separate terminal.
```
tensorboard --logdir runs
```


## Test
Generate pseudo labels.
 ```
python3 infer_cls.py --voc12_root /WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/img --network network.resnet38__cls --theta 0.2  --alpha 0.000001  --beta 0.99999 --sal_path /WeaklySupervisedSemanticSegmentation/Model/model_luad/thresHoldImages --weights /WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/5.pth --out_crf_pred /WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226


python3 infer_cls_demo_single.py --voc12_root /WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/img --network network.resnet38_PSHI_cls --theta 0.2  --alpha 0.000001  --beta 0.99999 --sal_path /WeaklySupervisedSemanticSegmentation/Model/model_luad/thresHoldImages --weights /WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/5.pth --out_crf_pred /WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226

# To generate result maps.

python3 infer_cls_demo.py --voc12_root /WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/img --network network.resnet38_PSHI_cls --theta 0.2  --alpha 0.000001  --beta 0.99999 --weights /WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/5.pth --sal_path /WeaklySupervisedSemanticSegmentation/Dataset/WSSS4LUAD/1.training/background-mask --out_crf_pred /WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226_wsss4luad --out_cam_pred /WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226_wsss4luad

 ```
 
Vis pseudo labels 
 ```
python3 colorful_luad.py --img_path /WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226 --out_path /WeaklySupervisedSemanticSegmentation/Model/model_luad/resnet38_luad_20221206_cam1step1ce/out_crf_pred25_copy13_20230226_wsss4luad
 ```


## Citation


