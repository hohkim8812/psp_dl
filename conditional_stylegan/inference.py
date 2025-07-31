import argparse
import torch
from torchvision import utils
from model import StyledGenerator
import os
import glob

def main():
    # Argument parser for command-line configuration
    parser = argparse.ArgumentParser(description='Inference with StyleGAN generator')
    parser.add_argument('--device', type=str, default='cuda')  # Device: cuda or cpu
    parser.add_argument('--labels', type=int, nargs='+', default=[0,1,2,3,4])  # List of labels for generation
    parser.add_argument('--n_per_label', type=int, default=120)  # Number of images to generate per label
    parser.add_argument('--step', type=int, default=5)  # Resolution step used in generator
    parser.add_argument('--save_dir', type=str, default='gen_images')  # Directory to save generated images
    parser.add_argument('--batch_size', type=int, default=100)  # Batch size for generation
    parser.add_argument('--ckpt_dir', type=str, default='checkpoint')  # Directory containing model checkpoints
    parser.add_argument('--ckpt_min', type=int, default=70000)  # Minimum checkpoint step to consider
    parser.add_argument('--ckpt_max', type=int, default=130000)  # Maximum checkpoint step to consider

    args = parser.parse_args()

    # Set device: use CUDA if available
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    # Load list of checkpoint files within the specified range
    ckpt_files = sorted(glob.glob(os.path.join(args.ckpt_dir, '[0-9]'*6 + '.model')))
    ckpt_files = [f for f in ckpt_files if args.ckpt_min <= int(os.path.basename(f)[:6]) <= args.ckpt_max]

    for ckpt_path in ckpt_files:
        # Extract step number from checkpoint filename
        step_str = os.path.basename(ckpt_path)[:6]
        ckpt_dir = os.path.join(args.save_dir, step_str)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Initialize generator and load weights from checkpoint
        generator = StyledGenerator(512).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            generator.load_state_dict(ckpt)
        except RuntimeError:
            generator.load_state_dict(ckpt['g_running'])  # Fallback in case of wrapped state dict
        generator.eval()

        for label in args.labels:
            # Directory to save images for this label
            label_dir = os.path.join(ckpt_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            count = 0

            # Generate images in batches
            for b in range(0, args.n_per_label, args.batch_size):
                current_bs = min(args.batch_size, args.n_per_label - b)
                z = torch.randn(current_bs, 512, device=device)  # Random latent vectors
                label_tensor = torch.full((current_bs,), label, dtype=torch.float32, device=device)  # Condition label

                with torch.no_grad():
                    imgs = generator(z, label_tensor, step=args.step, alpha=1)  # Generate images

                # Save each image in the appropriate label directory
                for img in imgs:
                    utils.save_image(img, os.path.join(label_dir, f"{count}.png"), normalize=True, value_range=(-1, 1))
                    count += 1

            print(f"Step {step_str}, Label {label}: Saved {count} images in {label_dir}")

# Entry point
if __name__ == '__main__':
    main()
