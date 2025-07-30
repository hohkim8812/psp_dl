from pathlib import Path
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare_data.py
image_dir = 'extended_tensile_test_images'  # Directory containing original .tif microstructure images
#어디에 있어도 바로바로 로드할 수 있게 만들어야함 
image_files = [f'{i}.tif' for i in range(1, 6)]  # List of image filenames
label_file = 'tensile_test_result.txt'  # File containing mechanical property data
crop_size = 128        # Crop size for image patches
stride = 32            # Stride used in sliding window cropping
crop_images = 'crop_images'

target_col = 0         # Target property column: 0=tensile, 1=yield, 2=elongation
group_size = 5
aug_per_group = 999    # Number of augmentations per group (random label assignment)

# dataset.py
test_size = 0.2            # Fraction of data used for testing
train_batch_size = 16      # Batch size during training
test_batch_size = 16       # Batch size during testing

# model.py and train.py
resnet_variant = 'resnet101'             # ResNet variant to use
pretrained_weights = 'IMAGENET1K_V1'     # Pretrained weights source
in_channels = 1                          # Grayscale image input (1 channel)

lr = 0.0002            # Learning rate
num_epochs = 70        # Total training epochs
model_dir = 'trained_pth' # 학습된 모델 저장 경로 

# inference.py
inference_image_dir = 'gen_images_2'     # Directory for generated microstructure images

