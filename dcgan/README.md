
# Conditional DCGAN for Microstructure Image Generation

### 1. INTRODUCTION
- This is a PyTorch-based Conditional DCGAN model for generating microstructure images under given conditions.

---

### 2. REQUIREMENTS

- The recommended python version is 3.10.0 The necessary Python packages are listed in requirements.txt.
- A CUDA-compatible GPU is recommended for faster training.

---

### 3. DESCRIPTION

#### 3.1 `config.py`
- Configuration file for paths and hyperparameters such as `image_dir`, `crop_size`, `stride`, `stride2`, learning rate, and output directories.

#### 3.2 `prepare_data.py`
- Prepares data for training by cropping original 1280x960 grayscale images into 128x128 pixel patches using specified strides.
- Augmented images are categorized by tempering temperature (500, 550, 600, 650, 700).

#### 3.3 `dataset.py`
- Constructs the dataset and dataloader using the augmented images.
- Dataset is split into training and test sets using a 9:1 ratio by default.

#### 3.4 `model.py`
- Defines the Conditional DCGAN architecture.
- The Generator takes a latent vector and label as input and generates 128x128 images.

#### 3.5 `train.py`
- Trains the DCGAN using the dataset and model.
- Saves the generator weights as `generator.pth` after training.

#### 3.6 `inference.py`
- Loads the trained Generator (`generator.pth`) and generates microstructure images based on a provided label.

#### 3.7 `evaluate.py`
- Calculates the FID score between real and generated images.

#### 3.8 `run.py`
- Provides a CLI to execute the entire pipeline (prepare, train, infer, evaluate).
- Uses the `fire` package for command-line interaction with commands: `prepare`, `train`, `inference`, `evaluate`, and `all`.

---

### 4. HOW TO USE

#### 4.1 Synopsis
```bash
python run.py <command>
```

#### 4.2 Commands

| Command     | Description                                                 |
|-------------|-------------------------------------------------------------|
| `prepare`   | Crop and augment images for training and evaluation         |
| `train`     | Train the DCGAN model using cropped images                  |
| `inference` | Generate images with the trained model                      |
| `evaluate`  | Evaluate generated images using the FID score               |

---

### 5. EXAMPLES

#### Generate training and evaluation data
```bash
python run.py prepare
```

#### Train the model
```bash
python run.py train
```

#### Generate images using the trained model
```bash
python run.py inference
```

#### Evaluate generated images
```bash
python run.py evaluate
```

#### Full pipeline (train + generate + evaluate)
```bash
python run.py all
```

#### Custom data directory and settings
```bash
python run.py prepare --image_dir="alt_data" --crop_size=96 --stride=48 --stride2=64
python run.py train --n_epochs=150 --g_lr=0.0001
python run.py inference --output_dir="gen_alt" --label_list="[500,550,600,650,700]" --num_images_per_label=120
python run.py evaluate --real_dir="real_resized_alt" --gen_dir="gen_resized_alt" --img_size="(150,150)"
```

#### Or all at once:
```bash
python run.py all --image_dir="alt_data" --crop_size=96 --stride=48 --stride2=64 --n_epochs=150 --g_lr=0.0001 --output_dir="gen_alt" --label_list="[500,550,600,650,700]" --num_images_per_label=120 --real_dir="real_resized_alt" --gen_dir="gen_resized_alt" --img_size="(150,150)"
```
