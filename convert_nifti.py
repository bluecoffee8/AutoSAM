# adopted from https://github.com/SLDGroup/MERIT/blob/main/utils/preprocess_synapse_data.py
# also check https://github.com/assassint2017/abdominal-multi-organ-segmentation/blob/master/dataset/dataset.py

import os
import shutil 
import imageio 
import nibabel as nib 
import numpy as np
from PIL import Image
import imageio.v3 as iio

MUL = 84

# 3,5,13,14,16,31,36,38,43,44,60,65,82,93

desired_png = './ACDC/imgs/patient_038_frame_01/frame_01_000.png'
png = iio.imread(desired_png)
# print(np.unique(png))
# print(png.shape)
# print(png)
# print(np.unique(png))
# print(png)
# png *= MUL
png *= 2
# iio.imwrite('./acdc_label1.png', png)
# print(np.unique(png))
# png *= 2
iio.imwrite('./acdc_example.png', png)
desired_png = './ACDC/annotations/patient_038_frame_01/frame_01_000.png'
png = iio.imread(desired_png)
png *= MUL 
iio.imwrite('./acdc_label.png', png)

folderpath = './output_experiment/novel5_120_5_original/'
# folderpath = './output_experiment/original_g120_5/'
inferpath = folderpath + 'infer/'
labelpath = folderpath + 'label/'

subject = 'patient_038.nii'
subject_infer = inferpath + subject 
subject_label = labelpath + subject

img = nib.load(subject_infer)
img_fdata = img.get_fdata()
# print(img_fdata.shape)
slice = 0
output = img_fdata[slice,:,:] * MUL
# output = np.stack((output, output, output), axis=-1)
output = output.astype(np.uint8)
iio.imwrite('./acdc_pred1.png', output)

folderpath = './output_experiment/novel5_120_5/'
inferpath = folderpath + 'infer/'
labelpath = folderpath + 'label/'
subject_infer = inferpath + subject 
subject_label = labelpath + subject
img = nib.load(subject_infer)
img_fdata = img.get_fdata()
slice = 0
output = img_fdata[slice,:,:] * MUL
output = output.astype(np.uint8)
iio.imwrite('./acdc_pred2.png', output)

folderpath = './output_experiment/original_g120_5/'
inferpath = folderpath + 'infer/'
labelpath = folderpath + 'label/'
subject_infer = inferpath + subject 
subject_label = labelpath + subject
img = nib.load(subject_infer)
img_fdata = img.get_fdata()
slice = 0
output = img_fdata[slice,:,:] * MUL
output = output.astype(np.uint8)
iio.imwrite('./acdc_pred3.png', output)

folderpath = './output_experiment/g120_5/'
inferpath = folderpath + 'infer/'
labelpath = folderpath + 'label/'
subject_infer = inferpath + subject 
subject_label = labelpath + subject
img = nib.load(subject_infer)
img_fdata = img.get_fdata()
slice = 0
output = img_fdata[slice,:,:] * MUL
output = output.astype(np.uint8)
iio.imwrite('./acdc_pred4.png', output)

# # print(output.shape)
# output_im = Image.fromarray(output, "RGB")
# output_im.save('./output_image.jpeg')
# print(np.sum(pred == 1))
# print(np.sum(pred == 2))
# print(np.sum(pred == 3))
# print(np.sum(pred == 0))

