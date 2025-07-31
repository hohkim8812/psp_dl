from pathlib import Path
import torch

# ------------------------------------------------------------------------------
# Device configuration
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically select GPU if available

# ------------------------------------------------------------------------------
# prepare_data.py parameters
# ------------------------------------------------------------------------------
image_dir = 'images'  # Directory containing original .tif microstructure images

image_files = [f'{i}.tif' for i in range(1, 6)]  # List of raw microstructure image filenames

label_file = 'tensile_test_result.txt'  # CSV or TXT file containing mechanical test results

crop_size = 128        # Size of each cropped patch (in pixels, square)
stride = 32            # Step size for sliding window cropping
crop_images = 'crop_images'  # Directory where cropped images will be saved

target_col = 0         # Column index in label file to predict: 0=tensile strength, 1=yield strength, 2=elongation
group_size = 5         # Number of rows in label file considered one group (used for label augmentation)
aug_per_group = 999    # Number of label augmentations per group (random sampling within group range)

# ------------------------------------------------------------------------------
# dataset.py parameters
# ------------------------------------------------------------------------------
test_size = 0.2            # Proportion of dataset to use for testing
train_batch_size = 16      # Batch size for training DataLoader
test_batch_size = 16       # Batch size for testing DataLoader

# ------------------------------------------------------------------------------
# model.py / train.py parameters
# ------------------------------------------------------------------------------
resnet_variant = 'resnet101'             # Backbone ResNet version (options: 'resnet18', 'resnet50', 'resnet101')
pretrained_weights = 'IMAGENET1K_V1'     # Pretrained weight source from torchvision
in_channels = 1                          # Number of input image channels (1 for grayscale microstructures)

lr = 0.0002            # Learning rate for optimizer
num_epochs = 70        # Total number of training epochs
model_dir = 'trained_pth'  # Directory to save trained model checkpoints

# ------------------------------------------------------------------------------
# inference.py parameters
# ------------------------------------------------------------------------------
inference_image_dir = 'gen_images_2'     # Directory containing generated microstructure images for inference
