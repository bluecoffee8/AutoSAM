import glob
import pickle

import logging
import os
import sys
import SimpleITK as sitk
import numpy as np
import argparse
from medpy import metric
import argparse
from scipy.ndimage import zoom
import torch
from einops import repeat
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset.Synapse import SynapseDataset

class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'pancreas'}

def compute_dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def calculate_metric_percase(pred, gt): 
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        # dice = metric.binary.dc(pred, gt)
        dice = compute_dice(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        # return dice, hd95
        return dice
    elif pred.sum() > 0 and gt.sum() == 0:
        # return 1, 0
        return 1
    else:
        # return 0, 0
        return 0

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0

def test_single_volume(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # print('IMAGE: ' + str(image.shape))
    # print('LABEL: ' + str(label.shape))

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != input_size[0] or y != input_size[1]:
                slice = zoom(slice, (input_size[0] / x, input_size[1] / y), order=3)  # previous using 0
            new_x, new_y = slice.shape[0], slice.shape[1]  # [input_size[0], input_size[1]]
            if new_x != patch_size[0] or new_y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / new_x, patch_size[1] / new_y), order=3)  # previous using 0, patch_size[0], patch_size[1]
            inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            # print('INPUTS: ' + str(inputs.shape))

            net.eval()
            with torch.no_grad():
                # outputs = net(inputs, multimask_output, patch_size[0])

                # outputs = net(inputs)
                # output_masks = outputs['masks']
                # out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                # out = out.cpu().detach().numpy()

                # mask = net(inputs) # shape (b, c (num_classes), h, w)
                # mask_softmax = F.softmax(mask, dim=1)
                # out = torch.argmax(mask_softmax, dim=1)
                # out = out.cpu().detach().numpy()

                mask = net(inputs)
                if isinstance(mask, tuple):
                    # print('TUPLE!')
                    # print(len(mask))
                    # print(mask[0].shape)
                    # print(mask[1].shape)
                    mask = mask[0]
                    mask = mask.squeeze(1).unsqueeze(0)
                    # print(mask.shape)

                # print('MASK: ' + str(mask.shape))

                out = torch.argmax(torch.softmax(mask, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()

                # print('OUT: ' + str(out.shape))

                out_h, out_w = out.shape
                if x != out_h or y != out_w:
                    pred = zoom(out, (x / out_h, y / out_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred
        # only for debug
        # if not os.path.exists('/output/images/pred'):
        #     os.makedirs('/output/images/pred')
        # if not os.path.exists('/output/images/label'):
        #     os.makedirs('/output/images/label')
        # assert prediction.shape[0] == label.shape[0]
        # for i in range(label.shape[0]):
        #     imageio.imwrite(f'/output/images/pred/pred_{i}.png', prediction[i])
        #     imageio.imwrite(f'/output/images/label/label_{i}.png', label[i])
        # temp = input('kkpsa')
    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list

def inference(args, multimask_output, db_config, model, test_save_path=None):
    db_test = db_config['Dataset'](base_dir='../SAMed/test_vol_h5', list_dir='../SAMed/lists/lists_Synapse', split='test_vol')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=8, multimask_output=multimask_output,
                                      patch_size=[1024, 1024], input_size=[224, 224],
                                      test_save_path=test_save_path, case=case_name, z_spacing=db_config['z_spacing'])
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)))
    metric_list = metric_list / len(db_test)
    for i in range(1, 8 + 1):
        try:
            logging.info('Mean class %d name %s mean_dice %f' % (i, class_to_name[i], metric_list[i - 1]))
        except:
            logging.info('Mean class %d mean_dice %f' % (i, metric_list[i - 1]))
    performance = np.mean(metric_list, axis=0)
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f' % (performance))
    logging.info("Testing Finished!")
    return 1

def test_brats(args):
    label_list = sorted(glob.glob(os.path.join(args.save_dir, 'label', '*nii')))
    infer_list = sorted(glob.glob(os.path.join(args.save_dir, 'infer', '*nii')))
    print("loading success...")
    Dice_et = []
    Dice_tc = []
    Dice_wt = []

    HD_et = []
    HD_tc = []
    HD_wt = []

    def process_label(label):
        net = label == 2
        ed = label == 1
        et = label == 3
        ET = et
        TC = net + et
        WT = net + et + ed
        return ET, TC, WT

    fw = open(args.save_dir + '/dice_pre.txt', 'a')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, infer = read_nii(label_path), read_nii(infer_path)
        label_et, label_tc, label_wt = process_label(label)
        infer_et, infer_tc, infer_wt = process_label(infer)
        Dice_et.append(dice(infer_et, label_et))
        Dice_tc.append(dice(infer_tc, label_tc))
        Dice_wt.append(dice(infer_wt, label_wt))

        HD_et.append(hd(infer_et, label_et))
        HD_tc.append(hd(infer_tc, label_tc))
        HD_wt.append(hd(infer_wt, label_wt))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('hd_et: {:.4f}\n'.format(HD_et[-1]))
        fw.write('hd_tc: {:.4f}\n'.format(HD_tc[-1]))
        fw.write('hd_wt: {:.4f}\n'.format(HD_wt[-1]))
        fw.write('*' * 20 + '\n', )
        fw.write('Dice_et: {:.4f}\n'.format(Dice_et[-1]))
        fw.write('Dice_tc: {:.4f}\n'.format(Dice_tc[-1]))
        fw.write('Dice_wt: {:.4f}\n'.format(Dice_wt[-1]))

        # print('dice_et: {:.4f}'.format(np.mean(Dice_et)))
        # print('dice_tc: {:.4f}'.format(np.mean(Dice_tc)))
        # print('dice_wt: {:.4f}'.format(np.mean(Dice_wt)))
    dsc = []
    avg_hd = []
    dsc.append(np.mean(Dice_et))
    dsc.append(np.mean(Dice_tc))
    dsc.append(np.mean(Dice_wt))

    avg_hd.append(np.mean(HD_et))
    avg_hd.append(np.mean(HD_tc))
    avg_hd.append(np.mean(HD_wt))

    fw.write('Dice_et' + str(np.mean(Dice_et)) + ' ' + '\n')
    fw.write('Dice_tc' + str(np.mean(Dice_tc)) + ' ' + '\n')
    fw.write('Dice_wt' + str(np.mean(Dice_wt)) + ' ' + '\n')

    fw.write('HD_et' + str(np.mean(HD_et)) + ' ' + '\n')
    fw.write('HD_tc' + str(np.mean(HD_tc)) + ' ' + '\n')
    fw.write('HD_wt' + str(np.mean(HD_wt)) + ' ' + '\n')

    fw.write('Dice' + str(np.mean(dsc)) + ' ' + '\n')
    fw.write('HD' + str(np.mean(avg_hd)) + ' ' + '\n')
    fw.close()
    with open(args.save_dir + '/dice_pre.txt', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            print(line)


def test_acdc(args):
    label_list = sorted(glob.glob(os.path.join(args.save_dir, 'label', '*nii')))
    infer_list = sorted(glob.glob(os.path.join(args.save_dir, 'infer', '*nii')))

    Dice_rv = []
    Dice_myo = []
    Dice_lv = []

    # hd_rv = []
    # hd_myo = []
    # hd_lv = []

    def process_label(label):
        rv = label == 1
        myo = label == 2
        lv = label == 3

        return rv, myo, lv

    fw = open(args.save_dir + '/dice_pre.txt', 'a')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label = read_nii(label_path)
        infer = read_nii(infer_path)
        label_rv, label_myo, label_lv = process_label(label)
        infer_rv, infer_myo, infer_lv = process_label(infer)

        Dice_rv.append(dice(infer_rv, label_rv))
        Dice_myo.append(dice(infer_myo, label_myo))
        Dice_lv.append(dice(infer_lv, label_lv))

        # hd_rv.append(hd(infer_rv, label_rv))
        # hd_myo.append(hd(infer_myo, label_myo))
        # hd_lv.append(hd(infer_lv, label_lv))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        # fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        # fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        # fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))
        # fw.write('*'*20+'\n')
        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_rv: {:.4f}\n'.format(Dice_rv[-1]))
        fw.write('Dice_myo: {:.4f}\n'.format(Dice_myo[-1]))
        fw.write('Dice_lv: {:.4f}\n'.format(Dice_lv[-1]))
        # fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        # fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        # fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))
        fw.write('*' * 20 + '\n')

    # fw.write('*'*20+'\n')
    # fw.write('Mean_hd\n')
    # fw.write('hd_rv'+str(np.mean(hd_rv))+'\n')
    # fw.write('hd_myo'+str(np.mean(hd_myo))+'\n')
    # fw.write('hd_lv'+str(np.mean(hd_lv))+'\n')
    # fw.write('*'*20+'\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_rv' + str(np.mean(Dice_rv)) + '\n')
    fw.write('Dice_myo' + str(np.mean(Dice_myo)) + '\n')
    fw.write('Dice_lv' + str(np.mean(Dice_lv)) + '\n')
    fw.write('Mean_HD\n')
    # fw.write('HD_rv' + str(np.mean(hd_rv)) + '\n')
    # fw.write('HD_myo' + str(np.mean(hd_myo)) + '\n')
    # fw.write('HD_lv' + str(np.mean(hd_lv)) + '\n')
    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_rv))
    dsc.append(np.mean(Dice_myo))
    dsc.append(np.mean(Dice_lv))
    avg_hd = []
    # avg_hd.append(np.mean(hd_rv))
    # avg_hd.append(np.mean(hd_myo))
    # avg_hd.append(np.mean(hd_lv))
    fw.write('avg_hd:' + str(np.mean(avg_hd)) + '\n')

    fw.write('DSC:' + str(np.mean(dsc)) + '\n')
    fw.write('HD:' + str(np.mean(avg_hd)) + '\n')

    print('done')
    fw.close()
    with open(args.save_dir + '/dice_pre.txt', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            print(line)

def test_kvasir(args):
    label_list = sorted(glob.glob(os.path.join(args.save_dir, 'label', '*nii')))
    infer_list = sorted(glob.glob(os.path.join(args.save_dir, 'infer', '*nii')))

    Dice_polyp = []

    def process_label(label):
        polyp = label == 1

        return polyp
    
    fw = open(args.save_dir + '/dice_pre.txt', 'a')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label = read_nii(label_path)
        infer = read_nii(infer_path)
        label_polyp = process_label(label)
        infer_polyp = process_label(infer)

        Dice_polyp.append(dice(infer_polyp, label_polyp))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_polyp: {:.4f}\n'.format(Dice_polyp[-1]))
        fw.write('*' * 20 + '\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_polyp' + str(np.mean(Dice_polyp)) + '\n')
    fw.write('Mean_HD\n')
    fw.write('*' * 20 + '\n')

    print('done')
    fw.close()
    with open(args.save_dir + '/dice_pre.txt', 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            print(line)
        

def test_synapse(args, model):
    if not os.path.exists(args.synapse_save_path):
        os.makedirs(args.synapse_save_path)
    test_save_path = os.path.join(args.synapse_save_path, 'predictions')
    os.makedirs(test_save_path, exist_ok=True)
    log_folder = os.path.join(args.synapse_save_path, 'test_log')
    os.makedirs(log_folder, exist_ok=True)

    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    inference(args=args, multimask_output=True, db_config={
            'Dataset': SynapseDataset,
            'volume_path': '../SAMed/test_vol_h5',
            'list_dir': '../SAMed/lists/lists_Synapse',
            'num_classes': 8,
            'z_spacing': 1
        }, model=model, test_save_path=test_save_path)

# def test_synapse(args):
#     label_list = sorted(glob.glob(os.path.join(args.save_dir, 'label', '*nii')))
#     infer_list = sorted(glob.glob(os.path.join(args.save_dir, 'infer', '*nii')))
#     Dice_spleen = []
#     Dice_right_kidney = []
#     Dice_left_kidney = []
#     Dice_gallbladder = []
#     Dice_esophagus = []
#     Dice_liver = []
#     Dice_stomach = []
#     Dice_aorta = []
#     Dice_inferior_vena_cava = []
#     Dice_portal_splenic_vein = []
#     Dice_pancreas = []
#     Dice_right_adrenal_gland = []
#     Dice_left_adrenal_gland = []

#     # hd_spleen = []
#     # hd_right_kidney = []
#     # hd_left_kidney = []
#     # hd_gallbladder = []
#     # hd_liver = []
#     # hd_stomach = []
#     # hd_aorta = []
#     # hd_pancreas = []

#     def process_label(label):
#         spleen = label == 1
#         right_kidney = label == 2
#         left_kidney = label == 3
#         gallbladder = label == 4
#         esophagus = label == 5
#         liver = label == 6
#         stomach = label == 7
#         aorta = label == 8
#         inferior_vena_cava = label == 9
#         portal_splenic_vein = label == 10
#         pancreas = label == 11
#         right_adrenal_gland = label == 12
#         left_adrenal_gland = label == 13

#         return spleen, right_kidney, left_kidney, gallbladder, esophagus, liver, stomach, aorta, inferior_vena_cava, portal_splenic_vein, pancreas, right_adrenal_gland, left_adrenal_gland

#     fw = open(args.save_dir + '/dice_pre.txt', 'a')
#     for label_path, infer_path in zip(label_list, infer_list):
#         print(label_path.split('/')[-1])
#         print(infer_path.split('/')[-1])
#         label, infer = read_nii(label_path), read_nii(infer_path)
#         label_spleen, label_right_kidney, label_left_kidney, label_gallbladder, label_esophagus, label_liver, label_stomach, label_aorta, label_inferior_vena_cava, label_portal_splenic_vein, label_pancreas, label_right_adrenal_gland, label_left_adrenal_gland = process_label(label)
#         infer_spleen, infer_right_kidney, infer_left_kidney, infer_gallbladder, infer_esophagus, infer_liver, infer_stomach, infer_aorta, infer_inferior_vena_cava, infer_portal_splenic_vein, infer_pancreas, infer_right_adrenal_gland, infer_left_adrenal_gland = process_label(infer)

#         Dice_spleen.append(dice(infer_spleen, label_spleen))
#         Dice_right_kidney.append(dice(infer_right_kidney, label_right_kidney))
#         Dice_left_kidney.append(dice(infer_left_kidney, label_left_kidney))
#         Dice_gallbladder.append(dice(infer_gallbladder, label_gallbladder))
#         Dice_esophagus.append(dice(infer_esophagus, label_esophagus))
#         Dice_liver.append(dice(infer_liver, label_liver))
#         Dice_stomach.append(dice(infer_stomach, label_stomach))
#         Dice_aorta.append(dice(infer_aorta, label_aorta))
#         Dice_inferior_vena_cava.append(dice(infer_inferior_vena_cava, label_inferior_vena_cava))
#         Dice_portal_splenic_vein.append(dice(infer_portal_splenic_vein, label_portal_splenic_vein))
#         Dice_pancreas.append(dice(infer_pancreas, label_pancreas))
#         Dice_right_adrenal_gland.append(dice(infer_right_adrenal_gland, label_right_adrenal_gland))
#         Dice_left_adrenal_gland.append(dice(infer_left_adrenal_gland, label_left_adrenal_gland))

#         fw.write('*' * 20 + '\n', )
#         fw.write(infer_path.split('/')[-1] + '\n')
#         fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
#         fw.write('Dice_right_kidney: {:.4f}\n'.format(Dice_right_kidney[-1]))
#         fw.write('Dice_left_kidney: {:.4f}\n'.format(Dice_left_kidney[-1]))
#         fw.write('Dice_gallbladder: {:.4f}\n'.format(Dice_gallbladder[-1]))
#         fw.write('Dice_esophagus: {:.4f}\n'.format(Dice_esophagus[-1]))
#         fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
#         fw.write('Dice_stomach: {:.4f}\n'.format(Dice_stomach[-1]))
#         fw.write('Dice_aorta: {:.4f}\n'.format(Dice_aorta[-1]))
#         fw.write('Dice_inferior_vena_cava: {:.4f}\n'.format(Dice_inferior_vena_cava[-1]))
#         fw.write('Dice_portal_splenic_vein: {:.4f}\n'.format(Dice_portal_splenic_vein[-1]))
#         fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))
#         fw.write('Dice_right_adrenal_gland: {:.4f}\n'.format(Dice_right_adrenal_gland[-1]))
#         fw.write('Dice_left_adrenal_gland: {:4f}\n'.format(Dice_left_adrenal_gland[-1]))

#         # hd_spleen.append(hd(infer_spleen, label_spleen))
#         # hd_right_kidney.append(hd(infer_right_kidney, label_right_kidney))
#         # hd_left_kidney.append(hd(infer_left_kidney, label_left_kidney))
#         # hd_gallbladder.append(hd(infer_gallbladder, label_gallbladder))
#         # hd_liver.append(hd(infer_liver, label_liver))
#         # hd_stomach.append(hd(infer_stomach, label_stomach))
#         # hd_aorta.append(hd(infer_aorta, label_aorta))
#         # hd_pancreas.append(hd(infer_pancreas, label_pancreas))

#         # fw.write('hd_spleen: {:.4f}\n'.format(hd_spleen[-1]))
#         # fw.write('hd_right_kidney: {:.4f}\n'.format(hd_right_kidney[-1]))
#         # fw.write('hd_left_kidney: {:.4f}\n'.format(hd_left_kidney[-1]))
#         # fw.write('hd_gallbladder: {:.4f}\n'.format(hd_gallbladder[-1]))
#         # fw.write('hd_liver: {:.4f}\n'.format(hd_liver[-1]))
#         # fw.write('hd_stomach: {:.4f}\n'.format(hd_stomach[-1]))
#         # fw.write('hd_aorta: {:.4f}\n'.format(hd_aorta[-1]))
#         # fw.write('hd_pancreas: {:.4f}\n'.format(hd_pancreas[-1]))

#         dsc = []
#         # HD = []
#         dsc.append(Dice_spleen[-1])
#         dsc.append((Dice_right_kidney[-1]))
#         dsc.append(Dice_left_kidney[-1])
#         dsc.append(np.mean(Dice_gallbladder[-1]))
#         dsc.append(Dice_esophagus[-1])
#         dsc.append(np.mean(Dice_liver[-1]))
#         dsc.append(np.mean(Dice_stomach[-1]))
#         dsc.append(np.mean(Dice_aorta[-1]))
#         dsc.append(Dice_inferior_vena_cava[-1])
#         dsc.append(Dice_portal_splenic_vein[-1])
#         dsc.append(np.mean(Dice_pancreas[-1]))
#         dsc.append(Dice_right_adrenal_gland[-1])
#         dsc.append(Dice_left_adrenal_gland[-1])
#         fw.write('DSC:' + str(np.mean(dsc)) + '\n')

#         # HD.append(hd_spleen[-1])
#         # HD.append(hd_right_kidney[-1])
#         # HD.append(hd_left_kidney[-1])
#         # HD.append(hd_gallbladder[-1])
#         # HD.append(hd_liver[-1])
#         # HD.append(hd_stomach[-1])
#         # HD.append(hd_aorta[-1])
#         # HD.append(hd_pancreas[-1])
#         # fw.write('hd:' + str(np.mean(HD)) + '\n')

#     fw.write('*' * 20 + '\n')
#     fw.write('Mean_Dice\n')
#     fw.write('Dice_spleen' + str(np.mean(Dice_spleen)) + '\n')
#     fw.write('Dice_right_kidney' + str(np.mean(Dice_right_kidney)) + '\n')
#     fw.write('Dice_left_kidney' + str(np.mean(Dice_left_kidney)) + '\n')
#     fw.write('Dice_gallbladder' + str(np.mean(Dice_gallbladder)) + '\n')
#     fw.write('Dice_esophagus' + str(np.mean(Dice_esophagus)) + '\n')
#     fw.write('Dice_liver' + str(np.mean(Dice_liver)) + '\n')
#     fw.write('Dice_stomach' + str(np.mean(Dice_stomach)) + '\n')
#     fw.write('Dice_aorta' + str(np.mean(Dice_aorta)) + '\n')
#     fw.write('Dice_inferior_vena_cava' + str(np.mean(Dice_inferior_vena_cava)) + '\n')
#     fw.write('Dice_portal_splenic_vein' + str(np.mean(Dice_portal_splenic_vein)) + '\n')
#     fw.write('Dice_pancreas' + str(np.mean(Dice_pancreas)) + '\n')
#     fw.write('Dice_right_adrenal_gland' + str(np.mean(Dice_right_adrenal_gland)) + '\n')
#     fw.write('Dice_left_adrenal_gland' + str(np.mean(Dice_left_adrenal_gland)) + '\n')

#     # fw.write('Mean_hd\n')
#     # fw.write('hd_spleen' + str(np.mean(hd_spleen)) + '\n')
#     # fw.write('hd_right_kidney' + str(np.mean(hd_right_kidney)) + '\n')
#     # fw.write('hd_left_kidney' + str(np.mean(hd_left_kidney)) + '\n')
#     # fw.write('hd_gallbladder' + str(np.mean(hd_gallbladder)) + '\n')
#     # fw.write('hd_liver' + str(np.mean(hd_liver)) + '\n')
#     # fw.write('hd_stomach' + str(np.mean(hd_stomach)) + '\n')
#     # fw.write('hd_aorta' + str(np.mean(hd_aorta)) + '\n')
#     # fw.write('hd_pancreas' + str(np.mean(hd_pancreas)) + '\n')

#     fw.write('*' * 20 + '\n')

#     dsc = []
#     dsc.append(np.mean(Dice_spleen))
#     dsc.append(np.mean(Dice_right_kidney))
#     dsc.append(np.mean(Dice_left_kidney))
#     dsc.append(np.mean(Dice_gallbladder))
#     dsc.append(np.mean(Dice_esophagus))
#     dsc.append(np.mean(Dice_liver))
#     dsc.append(np.mean(Dice_stomach))
#     dsc.append(np.mean(Dice_aorta))
#     dsc.append(np.mean(Dice_inferior_vena_cava))
#     dsc.append(np.mean(Dice_portal_splenic_vein))
#     dsc.append(np.mean(Dice_pancreas))
#     dsc.append(np.mean(Dice_right_adrenal_gland))
#     dsc.append(np.mean(Dice_left_adrenal_gland))
#     fw.write('dsc:' + str(np.mean(dsc)) + '\n')

#     # HD = []
#     # HD.append(np.mean(hd_spleen))
#     # HD.append(np.mean(hd_right_kidney))
#     # HD.append(np.mean(hd_left_kidney))
#     # HD.append(np.mean(hd_gallbladder))
#     # HD.append(np.mean(hd_liver))
#     # HD.append(np.mean(hd_stomach))
#     # HD.append(np.mean(hd_aorta))
#     # HD.append(np.mean(hd_pancreas))
#     # fw.write('hd:' + str(np.mean(HD)) + '\n')

#     print('done')
#     fw.close()
#     with open(args.save_dir + '/dice_pre.txt', 'r') as f:
#         lines = f.read().splitlines()
#         for line in lines:
#             print(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.save_dir = 'output_experiment/sam_unet_seg_ACDC_f0_tr_75'
    test_acdc(args)
