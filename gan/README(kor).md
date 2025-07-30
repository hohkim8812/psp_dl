# Conditional GAN for Microstructure Image Generation

## Abstract  
조건에 맞는 미세조직 이미지를 생성하기위한 PyTorch기반 Conditional GAN 모델입니다. 

---
## Requirements  

The following Python packages and versions are required to run this project successfully:

- Python 3.10
- torch==2.5.1
- torchvision==0.20.1
- numpy
- opencv-python==4.11.0.86
- scipy
- scikit-learn
- fire==0.7.0
- Pillow
- tensorboard==2.19.0
- tqdm==4.67.1
A CUDA-compatible GPU is recommended for faster training.
---
## Workflow
### config.py

### prepare_data.py
- 학습 하기전 데이터를 준비하는 과정으로 이미지를 잘라서 증강 및 저장하는 과정입니다.
- 원본 1280 x  960 Pixel의 마이크로이미지를 흑백이미지(채널1) 128x128 크기로 stride 만큽 이동하면서 잘라내고 저장한다 생성된 이미지는 학습용 폴더인 crop_images 와 FID score를 비교하기위해 crop_images_fid 2개로 템퍼링 온도에 해당하는 폴더(500,550,600,650,700)에 PNG 파일로 저장된다. 원본파일이 저장된 경로는 config.py에서 image_dir 을 통해 변경할 수 있으며 crop_size, stide,stride2 를 통하여 이미지의 크기 및 개수를 변경할 수 있고 crop_images, crop_images_fid를 통해 학습용 이미지와 평가용 이미지의 경로를 변경 할 수 있다.  

### dataset.py
- 증강된 데이터를 통하여 데이터셋과 데이터 로더를 구성하였다. 
- config.py 에 test_size 를 통해 학습/테스트 데이터셋을 조정 할 수 있으며 기본 값으로는 학습/테스트 데이터셋 9:1로 분할 되어 있으며 train_batch_size 와 test_batch_size 를 통해 배치사이즈를 변경 가능하다. 

### model.py 
- Conditional GAN 모델 구조를 정의합니다.
- Generator 는 latent vector와 조건라벨을 입력받아 3개의 Conv2d 와 upsampling 블록으로 128x128 Pixel 크기의 이미지를 생성합니다.
- Discriminator 는 조건 라벨을 받아, 2개의 Conv2d 블록과 Fully Connected Layer를 통해 진짜/가짜를 판별합니다.

### train.py
- 데이터셋과 모델(CGAN)을 불러와 학습을 진행합니다.
- 학습 주요 파라미터들은 latent_dim(Generator 입력 noise), n_classes(조건(라벨) 개수), n_epoch(총 학습수), d_lr(Discriminator 학습률), g_lr(Generator 학습률), d_beta1, d_beta2, g_beta1, g_beta2(옵티마이저의 베타 파라미터)
등으로 config.py에서 조정하거나 명령어 --옵션명 =값 형태로 입력하여 조정 가능하다. (ex : python run.py train --n_epochs=500 --g_lr=0.0001 --latent_dim=128)
- 학습이 완료될시 모델의 가중치는 generator.pth로 저장됩니다.

### inference.py
- 학습된 모델(generator.pth) 를 불러와, 지정된 조건별로 미세조직 이미지를 생성합니다.
- 주요 파라미터는 num_images_per_label(라벨별로 생성할 이미지 수), label_list(생성할 라벨 목록), output_dir(생성 이미지가 저장될 폴더), latent_dim(Generator 입력 noise)로 config.py 에서 조정하거나 실행시 명령어 인자로 지정할 수 있다 (ex: python run.py inference --num_images_per_label=150 --output_dir="genator_gan"
).

### evaluate.py
- 실제 이미지와 생성된 이미지를 비교하여 FID 스코어를 계산합니다. 
- 평가를 위해 Real/GAN 이미지 모두 299x299크기로 리사이즈되고, 정규화되어 Inception 네트워크에 입력됩니다.
- real_dir(실제 이미지 경로), gen_dir(생성이미지 경로), img_size(리사이즈 이미지 크기)등을 config.py에서 조정하거나 명령어 인자로 지정할 수 있다 (ex: python run.py evaluate --real_dir="real_resized" --gen_dir="gen_resized_cgan").

### run.py
- 전체 데이터 준비, 학습, 이미지 생성, 평가 까지 한번에 실행할 수 있도록 CLI를 제공 합니다.
- fire 패키지를 사용하여 간단한 명력어로 파이프 라인을 제어할 수 있으며 주요 명령어는 prpare, train, inference, evaluate, all으로 용도에 맞게 선택적으로 실행할 수 있고 각 단계별 파라미터는 config.py에서 수정하거나 실행시 명령어로 전달가능 합니다.

## How to use

### SYNOPSIS
python run.py <command>

### DESCRIPTION

### COMMANDS

|    Command        |                   Description                         |
| ----------------- | ------------------------------------------------------|
| `prepare`         | 이미지 잘라서 증강 및 학습,평가용 이미지 저장 명령어    |
| `train`           | 크롭된 이미지와 모델(CGAN)으로 학습하는 명령어         |
| `inference`       | 학습된 모델을 통해 이미지를 생성하는 명령어             |
| `evaluate`        | 생성된 이미지를 FID를 통해 평가하는 명령어              |

## 실행 예제
### 학습 및 평가용 데이터 생성시
python run.py prepare
### 모델 학습시
python run.py train
### 학습된 모델을 통해 이미지 생성 
python run.py inference
### 생성된 이미지를 평가
python run.py evaluate
### 모델 학습,생성과 평가
python run.py all
### 새로운 데이터 경로 및 설정값으로 실행할 경우
python run.py prepare --image_dir="alt_data" --crop_size=96 --stride=48 --stride2=64
python run.py train --n_epochs=150 --g_lr=0.0001
python run.py inference --output_dir="gen_alt" --label_list="[500,550,600,650,700]" --num_images_per_label=120
python run.py evaluate --real_dir="real_resized_alt" --gen_dir="gen_resized_alt" --img_size="(150,150)"
으로 개별적으로 실행을 하거나 
python run.py all --image_dir="alt_data" --crop_size=96 --stride=48 --stride2=64 --n_epochs=150 --g_lr=0.0001 --output_dir="gen_alt" --label_list="[500,550,600,650,700]" --num_images_per_label=120 --real_dir="real_resized_alt" --gen_dir="gen_resized_alt" --img_size="(150,150)"