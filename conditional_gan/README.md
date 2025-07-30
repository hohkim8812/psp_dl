
# Conditional GAN for Microstructure Image Generation

### 1. INTRODUCTION
- This is a PyTorch-based Conditional GAN model for generating microstructure images under given conditions.

---

### 2. REQUIREMENTS

- The recommended python version is 3.10.0 The necessary Python packages are listed in requirements.txt.
- A CUDA-compatible GPU is recommended for faster training.

---

### 3. DESCRIPION

#### 3.1 `config.py`
- Configuration file for paths and hyperparameters such as image_dir, crop_size, stride, learning rate, and model variants.

#### 3.2 `prepare_data.py`
- Prepares the data for training by cropping and augmenting images.
- The cropped images are saved in two folders (`crop_images` for training and `crop_images_fid` for FID evaluation), each with subfolders for different tempering temperatures (500, 550, 600, 650, 700).

#### 3.3 `dataset.py`
- Constructs the dataset and dataloader using the augmented data.

#### 3.4 `model.py`
- Defines the structure of the Conditional GAN model.

#### 3.5 `train.py`
- Trains the Conditional GAN model using datasets and parameters.
- Trained Generator weights are saved as `generator.pth`.

#### 3.6 `inference.py`
- Loads the trained generator and generates microstructure images based on specified labels.

#### 3.7 `evaluate.py`
- Evaluates the similarity between real and generated images using the FID score.
- Both image sets are resized to 299x299 and normalized for input to the Inception network.

#### 3.8 `run.py`
- Provides a CLI interface to run the entire pipeline (prepare, train, infer, evaluate).
- Built with the `fire` package for simplified usage.

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
| `train`     | Train the CGAN model using cropped images                   |
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
