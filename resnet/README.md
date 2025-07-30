# Microstructure-Property Modeling: ResNet-101

## 1. INTRODUCTION
This project implements a PyTorch-based regression model that predicts mechanical properties—tensile strength, yield strength, and elongation—from microstructure images using the ResNet architecture.

---

## 2. REQUIREMENTS

The recommended python version is 3.10.0
The necessary Python packages are listed in requirements.txt
A CUDA-compatible GPU is recommended for faster training.

---

## 3. DESCRIPTION

### 3.1 `config.py`
- Configuration file for paths and hyperparameters such as `image_dir`, `crop_size`, `stride`, learning rate, and model variants.

### 3.2 `prepare_data.py`
- Prepares the data before training by augmenting and saving images and property values.
- Original microstructure images of size 1280x960 are cropped into grayscale (1-channel) images of size 128x128 using a stride of 32.
- The cropped images are saved as PNG files in temperature-specific folders (e.g., 500, 550, 600...).
- You can modify image path, crop size, or stride via `config.py`.
- Property labels are read and matched with images by generating random values within the min-max range for each group.

### 3.3 `dataset.py`
- Constructs datasets and dataloaders from the preprocessed data.
- Splits the dataset into training and test sets (8:2 ratio).
- Returns normalization parameters (mean, std) for later denormalization when computing RMSE and R².

### 3.4 `model.py`
- Defines the ResNet-101 regression model for property prediction from images.
- Model variants (resnet18, resnet50) can be selected via the `resnet_variant` function in `config.py`.
- Input: `[N, 1, 128, 128]` grayscale images  
  Output: predicted material properties.

### 3.5 `train.py`
- Trains the model using datasets and configuration from `dataset.py` and `model.py`.
- Parameters such as learning rate, number of epochs, target property, and model save path are configurable in `config.py`.
- Uses MSE as the loss function; evaluates performance using RMSE and R².
- After training, the model weights (e.g., `model0.pth` for tensile strength) and prediction results (`predictions_vs_actual_col0.xlsx`) are saved.

### 3.6 `inference.py`
- Loads trained models (`your_model.pth`) and predicts properties for given images.
- Predictions are saved in Excel format (e.g., `generator_property_prediction_col1.xlsx`).
- Each index represents (0: tensile, 1: yield, 2: elongation).

### 3.7 `run.py`
- Wraps the entire pipeline into a simple command-line interface using the `fire` package.
- Provides commands like `prepare`, `train`, `inference`, and `both` for streamlined execution.

---

## 4. HOW TO USE

### 4.1 Synopsis

```bash
python run.py <command> [--target=TARGET_COL]
```

### 4.2 Commands and flags

| Command        | Description                                                         |
|----------------|---------------------------------------------------------------------|
| `prepare_data` | Preprocess and augment microstructure and property data             |
| `train`        | Train model using cropped microstructure images                     |
| `inference`    | Predict properties from images using a trained model                |
| `both`         | Run both training and inference in one go                           |


| Flag                  | Description                               |
|-----------------------|-------------------------------------------|
| `--target=TARGET_COL` | Target property index (default: `0`)      |
|                       |    `0`: Tensile Strength                   |
|                       |    `1`: Yield Strength                     |
|                       |    `2`: Elongation                         |

## 5. EXAMPLE

### 5.1 Prepare data for training with target=0 (Tensile Strength)
```bash
python run.py prepare_data --target=0
```

### 5.2 Train with existing data (Tensile Strength)
```bash
python run.py train --target=0
```

### 5.3 Predict using a trained model (Tensile Strength)
```bash
python run.py inference --target=0
```

### 5.4 Run both training and inference (Tensile Strength)
```bash
python run.py both --target=0
```

### 5.5 Use alternative data path and settings
```bash
python run.py prepare --target=2 --image_dir="alt_data" --crop_size=96 --stride=48
```

```bash
python run.py train --target=2 --image_dir="alt_data" --crop_size=96 --stride=48
```

### or
```bash
python run.py both --target=2 --image_dir="alt_data" --crop_size=96 --stride=48
```