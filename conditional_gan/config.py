from pathlib import Path


# ===============================
# prepare_data.py related config
# ===============================

image_dir = "../images"  # Directory containing original raw images
crop_size = 128          # Size (in pixels) of square crop window
image_size = 128         # Expected size of full image (assumed square)
image_files = [f'{i}.tif' for i in range(1, 6)]  # List of image filenames to process

stride = 32              # Stride for standard cropping (dense)
stride2 = 96             # Stride for sparse cropping (e.g., for FID evaluation)
crop_images = 'crop_images'  # Folder to store standard cropped images
crop_images_fid = 'crop_images_fid'  # Folder to store FID-compatible cropped images


# ============================
# dataset.py related config
# ============================

test_size = 0.2              # Ratio of test set in train/test split
train_batch_size = 32        # Batch size for training DataLoader
test_batch_size = 32         # Batch size for testing DataLoader
latent_dim = 100             # Latent vector dimension (for GAN or generator input)
n_classes = 5                # Number of discrete class labels (e.g., for conditional GAN)


# ==========================
# train.py related config
# ==========================

n_epochs = 200               # Total number of training epochs

# Optimizer parameters for discriminator
d_lr = 0.0002
d_beta1 = 0.5
d_beta2 = 0.999

# Optimizer parameters for generator
g_lr = 0.0002
g_beta1 = 0.5
g_beta2 = 0.999


# ==============================
# inference.py related config
# ==============================

num_images_per_label = 108       # Number of images to generate per label
label_list = [500, 550, 600, 650, 700]  # List of label values (e.g., tempering temperatures)
output_dir = "gen_cgan"          # Directory to save generated images


# ==============================
# evaluate.py related config
# ==============================

n_per_label = 108               # Number of real/generated images per label used in FID
real_src_dir = "crop_images_fid"        # Source folder for real images (pre-FID resize)
real_dir = "real_resized"               # Folder to store resized real images
gen_src_dir = "gen_cgan"                # Source folder for generated images
gen_dir = "gen_resized_cgan"            # Folder to store resized generated images

img_size = (299, 299)           # Size to resize images for Inception-based FID
batch_size = 32                 # Batch size for FID feature extraction
seed = 42                       # Random seed for reproducibility
