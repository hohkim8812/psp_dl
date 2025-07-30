import fire
import config
from prepare_data import run_prepare_data
from train import main as train_main
from inference import main as inference_main
from evaluate import main as evaluate_main


def prepare_data():
    run_prepare_data()

def train(
    n_epochs=config.n_epochs,
    d_lr=config.d_lr,
    g_lr=config.g_lr,
    d_beta1=config.d_beta1,
    d_beta2=config.d_beta2,
    g_beta1=config.g_beta1,
    g_beta2=config.g_beta2,
    latent_dim=config.latent_dim,
    n_classes=config.n_classes,
    batch_size=config.train_batch_size,
    image_size=config.image_size
):
    train_main(n_epochs, d_lr, g_lr, d_beta1, d_beta2, g_beta1, g_beta2,
               latent_dim, n_classes, batch_size, image_size)

def inference(
    num_images_per_label=config.num_images_per_label,
    label_list=config.label_list,
    output_dir=config.output_dir,
    latent_dim=config.latent_dim
):
    inference_main(num_images_per_label, label_list, output_dir, latent_dim)

def evaluate(
    real_src_dir=config.real_src_dir,
    real_dir=config.real_dir,
    gen_src_dir=config.gen_src_dir,
    gen_dir=config.gen_dir,
    img_size=config.img_size,
    batch_size=config.batch_size,
    n_per_label=config.n_per_label,      
    seed=config.seed
):
    evaluate_main(real_src_dir, real_dir, gen_src_dir, gen_dir, img_size, batch_size, n_per_label, seed)

def all_steps():
    """Run all steps in order: prepare, train, inference, evaluate"""
    print("== Step 1: Prepare Data ==")
    prepare_data()
    print("== Step 2: Train Model ==")
    train()
    print("== Step 3: Inference ==")
    inference()
    print("== Step 4: Evaluate ==")
    evaluate()

if __name__ == "__main__":
    fire.Fire({
        "prepare": prepare_data,
        "train": train,
        "inference": inference,
        "evaluate": evaluate,
        "all": all_steps  
    })