from pathlib import Path


#prepare_data.py structure
image_dir=  "../extended_tensile_test_images" 
crop_size = 128
image_size= 128
image_files = [f'{i}.tif' for i in range(1, 6)]
#crop_size_M = ' ' 
#crop_size_N = ' ' 
stride = 32
stride2= 96
crop_images = 'crop_images'
crop_images_fid ='crop_images_fid'

#dataset.py parameter
test_size= 0.2
train_batch_size = 32
test_batch_size = 32
latent_dim=100
n_classes=5

#train.py parameter
n_epochs = 200
d_lr = 0.0002
g_lr = 0.0002
d_beta1 = 0.5
g_beta1 = 0.5
d_beta2 = 0.999
g_beta2 = 0.999

#inference.py parameter
num_images_per_label = 108
label_list = [500, 550, 600, 650, 700]
output_dir = "gen_cgan" # inference result

#evaluate.py parameter
n_per_label = 108
real_src_dir = "crop_images_fid"
real_dir    = "real_resized"
gen_src_dir = "gen_cgan"
gen_dir     = "gen_resized_cgan"
img_size = (299, 299)
batch_size = 32
seed = 42