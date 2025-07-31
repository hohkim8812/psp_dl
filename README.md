
# Modeling the relationship between tempering process-microstructure-property of steel using deep learning

This repository provides source code accompanying our research on modeling the **Process–Structure–Property (PSP)** relationships in 42CrMo4 using **deep learning**. The core objective of this study is to explore and implement generative and predictive models that can capture the complex transformations from processing parameters to microstructure, and subsequently from microstructure to mechanical properties.

> 📌 This work is currently under peer review.

---

### 📘 Overview

Understanding PSP relationships is a cornerstone of materials design and optimization. In this repository:

- **Conditional Generative Models** (e.g., Conditional GAN, DCGAN, StyleGAN) are employed to model the **Process → Structure** relationship.
- **ResNet-101** is used to learn the **Structure → Property** mapping via regression.

These models aim to serve as building blocks for end-to-end learning pipelines capable of generating microstructures or predicting performances.

---

### 📁 Directory Structure

| Directory                | PSP Link                | Description |
|--------------------------|-------------------------|-------------|
| `conditional_gan/`       | Process → Structure     | Implements a Conditional GAN to synthesize microstructure images conditioned on tempering temperature. |
| `conditional_dcgan/`     | Process → Structure     | Implements a Conditional DCGAN to synthesize microstructure images conditioned on tempering temperature. |
| `conditional_stylegan/`  | Process → Structure     | Implements a Conditional StyleGAN to synthesize microstructure images conditioned on tempering temperature. |
| `resnet/`                | Structure → Property    | Implements a ResNet101 to predict material properties (tensile strength, yield strength, elongation) given microstructure images. |

---

### 🚀 Getting Started

To reproduce results or run models, each directory includes:

- `config.py` for configuration
- `prepare_data.py` for data preprocessing
- `train.py` and `inference.py` scripts
- `evaluate.py` for quantitative evaluation (e.g., FID scores)
- `run.py` for streamlined command-line execution

Please refer to each folder's README or documentation for detailed instructions.

---

### 📊 Dependencies

- Python 3.10+
- PyTorch (>=2.0)
- torchvision, fire, Pillow, OpenCV, scikit-learn, TensorBoard, tqdm, scipy
- CUDA-enabled GPU is strongly recommended for training generative models

---

### 📄 Citation

If you find this repository useful in your research, please cite our work (under review):

---

### 📬 Contact

For questions, contributions, or collaboration inquiries, please feel free to open an issue or contact:

- **Hoheok Kim**  
- GitHub: [@hohkim8812](https://github.com/hohkim8812)  
- Email: hoheokkim@kims.re.kr

---
