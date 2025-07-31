from pathlib import Path

# ==============================================================================
# prepare_data.py settings
# ==============================================================================

image_dir = "images"  # Directory containing raw .tif microstructure images
crop_size = 128     # Size of square patch (in pixels) to crop from raw images
image_size = 128    # Target image size used for generation and evaluation
image_files = [f'{i}.tif' for i in range(1, 6)]  # List of image filenames (1.tif ~ 5.tif)

stride = 32         # Stride for training data cropping (more overlapping patches)
stride2 = 96        # Stride for FID evaluation cropping (less overlap)

crop_images = 'crop_images'  # Directory to save cropped patches for training
crop_images_fid = 'crop_images_fid'  # Directory for FID real image patches

# ==============================================================================
# dataset.py settings
# ==============================================================================

test_size = 0.2              # Ratio of test data split (20%)
train_batch_size = 32        # Batch size during training
test_batch_size = 32         # Batch size during evaluation
latent_dim = 100             # Size of latent vector (input to generator)
n_classes = 5                # Number of class labels (e.g., 500, 550, ..., 700)

# ==============================================================================
# train.py settings
# ==============================================================================

n_epochs = 200       # Total number of training epochs
d_lr = 0.0002        # Discriminator learning rate
g_lr = 0.0002        # Generator learning rate

# Adam optimizer parameters for discriminator
d_beta1 = 0.5
d_beta2 = 0.999

# Adam optimizer parameters for generator
g_beta1 = 0.5
g_beta2 = 0.999

# ==============================================================================
# inference.py settings
# ==============================================================================

num_images_per_label = 108     # Number of images to generate per label during inference
label_list = [500, 550, 600, 650, 700]  # Target labels for generation
output_dir = "gen_dcgan"       # Output directory for generated images

# ==============================================================================
# evaluate.py settings
# ==============================================================================

n_per_label = 108              # Number of real/generated images per label for FID computation
real_src_dir = "crop_images_fid"      # Source directory for real (cropped) patches
real_dir = "real_resized"            # Directory for resized real images
gen_src_dir = "gen_dcgan"            # Source directory for generated images
gen_dir = "gen_resized_dcgan"        # Directory for resized generated images
img_size = (299, 299)                # Image size expected by InceptionV3 for FID
batch_size = 32                      # Batch size for FID computation
seed = 42                            # Random seed for reproducibility
