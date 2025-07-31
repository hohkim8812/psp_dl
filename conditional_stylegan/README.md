
# Style-Based GAN for Microstructure Image Generation

### 1. INTRODUCTION
- This is a PyTorch-based StyleGAN model for generating microstructure images conditioned on process parameters.

---

### 2. WORKFLOW

#### 2.1 `prepare_data.py`
- Crops original 1280x960 microstructure images into 128x128 size patches and saves them.
- Converts the cropped images into LMDB datasets for efficient training.
- Trains StyleGAN using a Progressive Growing approach from the LMDB dataset.

**Key arguments:**
- `--folder_path`: Directory of original images
- `--output_root`: Directory to save cropped images for training
- `--output_root2`: Directory to save images for FID evaluation
- `--crop_size`: Cropping size
- `--imgout`: Path to output LMDB image data
- `--labelout`: Path to output LMDB label data
- `--n_worker`: Number of parallel data workers

#### 2.2 `dataset.py`
- Implements `MultiResolutionDataset` to load images and labels from LMDB.
- Supports loading at multiple resolutions.

#### 2.3 `model.py`
- Defines StyleGAN generator and discriminator with modules such as Progressive Growing, Noise Injection, and AdaIN.

#### 2.4 `projector.py`
- Projects input images into StyleGAN's latent space to find latent vectors that can reconstruct the given images.

**Key arguments:**
- `--ckpt`: Path to trained model checkpoint
- `--size`: Output image size
- `--files`: List of input images to project

#### 2.5 `train.py`
- Trains StyleGAN with Progressive Growing using the LMDB dataset.
- Supports options for style mixing, different loss functions (WGAN-GP or R1), and dynamic resolution.

**Key arguments:**
- `--img_path`: LMDB path for training images
- `--label_path`: LMDB path for training labels
- `--mixing`: Enable style mixing
- `--loss`: Loss function (`wgan-gp` or `r1`)
- `--max_size`: Maximum training resolution (e.g., 128 for 128×128)

#### 2.6 `inference.py`
- Loads trained generator checkpoints and generates images conditioned on given labels.
- Saves images in batches under corresponding label folders.

**Key arguments:**
- `--ckpt_dir`: Path to generator checkpoint
- `--labels`: List of labels to generate
- `--n_per_label`: Number of images per label
- `--batch_size`: Batch size for generation
- `--save_dir`: Base directory to save generated images

#### 2.7 `evaluate.py`
- Computes FID scores for generated images.
- Finds the best performing checkpoint and evaluates ferrite fraction.
- Outputs results in `.xlsx` and `.txt` formats.

**Key arguments:**
- `--real_src_dir`: Original real image directory (`images_fid`)
- `--real_dir`: Directory for resized real images (`images_fid_resized`)
- `--gen_src_dir_base`: Base directory of generated images (`gen_images`)
- `--gen_dir_base`: Base directory of resized generated images (`gen_images_resized`)
- `--img_size`: Resize image size (e.g., [299, 299])
- `--batch_size`: Batch size for Inception V3 input (default: 32)
- `--n_per_label`: Number of images per label (e.g., 108)
- `--ckpt_steps`: List of checkpoints to evaluate (e.g., 070000 to 130000)

---

### 3. HOW TO USE

#### 3.1 Synopsis
```bash
python prepare_data.py <args>
python train.py <args>
python inference.py <args>
python evaluate.py <args>
```

---

### 4. EXAMPLES

#### Crop images and create LMDB datasets
```bash
python prepare_data.py --folder_path images --output_root images/128x128 --output_root2 images_fid --img_src_path images --imgout images_lmdb --labelout labels_lmdb --n_worker 4
```

#### Train StyleGAN from LMDB
```bash
python train.py --img_path images_lmdb --label_path labels_lmdb --mixing --loss wgan-gp --max_size 128
```

#### Generate images using trained model
```bash
python inference.py --ckpt_dir checkpoint --ckpt_min 70000 --ckpt_max 130000 --labels 0 1 2 3 4 --n_per_label 120 --batch_size 100
```

#### Evaluate generated images
```bash
python evaluate.py --real_src_dir images_fid --real_dir images_fid_resized --gen_src_dir_base gen_images --gen_dir_base gen_images_resized --img_size 299 299 --batch_size 32 --n_per_label 108 --seed 42 --ckpt_steps 070000 080000 090000 100000 110000 120000 130000 --phase_fraction
```
