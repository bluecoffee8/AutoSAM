# AutoSAM 
This repo is pytorch implementation of paper "How to Efficiently Adapt Large Segmentation Model(SAM) to Medical Image Domains" by Xinrong Hu et al.

[[`Paper`](https://arxiv.org/pdf/2306.13731.pdf)]

![](./autosam.png)
## Setup
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. 

clone the repository locally:

```
git clone git@github.com:xhu248/AutoSAM.git
```
and install requirements:
```
cd AutoSAM; pip install -e .
```
Download the checkpoints from [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) and place them under AutoSAM/

## Dataset

The original ACDC data files can be dowonloaded at [Automated Cardiac Diagnosis Challenge ](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
The data is provided in nii.gz format. We convert them into PNG files as SAM requires RGB input. 
The processed data can be downloaded [here](https://drive.google.com/drive/folders/1RcpWYJ7EkwPiCR9u6HRrg7JHQ_Dr7494?usp=drive_link)

## How to use
### Finetune CNN decoder
```
python scripts/main_feat_seg.py --src_dir ${ACDC_folder} \
--data_dir ${ACDC_folder}/imgs/ --save_dir ./${output_dir}  \
--b 4 --dataset ACDC --gpu ${gpu} \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4
```
${tr_size} decides how many volumes used in the training; ${model_type} is selected from vit_b (default), vit_l, and vit_h;

### Finetune AutoSAM
```
python scripts/main_autosam_seg.py --src_dir ${ACDC_folder} \
--data_dir ${ACDC_folder}/imgs/ --save_dir ./${output_dir}  \
--b 4 --dataset ACDC --gpu ${gpu} \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4
```
This repo also supports distributed training
```
python scripts/main_autosam_seg.py --src_dir ${ACDC_folder} --dist-url 'tcp://localhost:10002' \
--data_dir ${ACDC_folder}/imgs/ --save_dir ./${output_dir} \
--multiprocessing-distributed --world-size 1 --rank 0  -b 4 --dataset ACDC \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4
```
## Todo
* Evaluate on more datasets
* Add more baselines

## Citation
If you find our codes useful, please cite
```
@article{hu2023efficiently,
  title={How to Efficiently Adapt Large Segmentation Model (SAM) to Medical Images},
  author={Hu, Xinrong and Xu, Xiaowei and Shi, Yiyu},
  journal={arXiv preprint arXiv:2306.13731},
  year={2023}
}
```

# Additional Notes by Kevin Li

## Basic Setup

1. Clone repository 
2. Follow setup steps for installing libraries (medsam conda env) above. In addition, need to separately install tensorboard (via conda), batchgenerators, and medpy (both via pip). 
3. Repo for MedSAM (forked) is here: https://github.com/bluecoffee8/MedSAM/tree/main
4. Download medsam_vit_b model checkpoint from https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN and place in AutoSAM/ folder.
5. Ensure that in main_feat_seg and main_autosam_seg, in main_worker() function, the case where args.model_type == 'vit_b', the model_checkpoint should be 'medsam_vit_b.pth'
6. Download ACDC data from https://drive.google.com/drive/folders/1RcpWYJ7EkwPiCR9u6HRrg7JHQ_Dr7494 and unzip annotations.zip and imgs.zip. Create an ACDC/ folder within AutoSAM/ folder and place splits.pkl, annotations, and imgs inside ACDC/

## Novel Experiments

### Finetune CNN or ViT
Project Overview
As a refresher, the novel experiments I will be running involving fine-tuning the prediction head (whether it is MLP, CNN, or ViT), with image encoder weights frozen. In the original repo (https://github.com/xhu248/AutoSAM) for the paper 'How to Efficiently Adapt Large Segmentation Model (SAM) to Medical Images' by Xinrong Hu, Xiaowei Xu, and Yiyu Shi (2023), the authors used original SAM image encoder weights. The novel experiments I propose involve using MedSAM image encoder weights, from the paper 'Segment Anything in Medical Images' by Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang (2023). The authors of the MedSAM paper provide weights for a SAM vit_b type model, so one simply needs to download this model checkpoint and make minor modifications to original codebase to finetune using these new image encoder weights. The bulk of the fine-tuning code is already written and copied over from original repo. 

Instructions
1. In scripts folder, in 'main_feat_seg' (for CNN) or 'main_autosam_seg' (for ViT), in function save_checkpoint() change the location to where best model is copied to desired path. Also near the end of main_worker() make sure the code block involving save_checkpoint() is uncommented. This ensures that the best model over all epochs is saved somewhere. 
2. Simply run finetune_cnn.sh file for CNN (finetune_vit.sh for ViT). For input arguments, if running on 1 GPU, ensure --workers=1. Finetune for 120 epochs. We use 1 GPU so ensure --gpu=0. Ensure --model_type=vit_b, --src_dir=ACDC/, --data_dir=ACDC/imgs/, --classes=4, --num_classes=4. So far we run experiment only on fold 0, so --fold=0. One can run two kinds of experiments, one with --tr_size=1 and other with --tr_size=5 (size of volume to use during training). --save_dir should equal the name of folder within AutoSAM/output_experiment where you wish to store results. Ensure --dataset=ACDC. 

### Finetune MLP
1. In models folder, in 'build_sam_feat_seg_model.py', in '_build_feat_seg_model', uncomment the MLP model for SegDecoder and comment out CNN model. Then just run finetune_cnn.sh. 
2. Simply remember to change the best_model path in 'main_feat_seg' save_checkpoint() function, and also make sure --save_dir is changed to folder you wish to save results in. Also ensure that tr_size is the volume size you want (either 1 or 5). 

### Evalute on Best Model

1. In the test.py file, uncomment the correct model ('sam_feat_seg_model_registry' for MLP/CNN, 'sam_seg_model_registry' for ViT). Load the correct checkpoint (have a path to the desired best model checkpoint file), for MLP/CNN use test_cnn, else if ViT use test_vit. 
2. Simply run run_test.sh.