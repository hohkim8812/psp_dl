import argparse
import torch
from torchvision import utils
from model import StyledGenerator
import os
import glob

def main():
    parser = argparse.ArgumentParser(description='Inference with StyleGAN generator')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--labels', type=int, nargs='+', default=[0,1,2,3,4])
    parser.add_argument('--n_per_label', type=int, default=120)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='gen_images')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoint')
    parser.add_argument('--ckpt_min', type=int, default=70000)
    parser.add_argument('--ckpt_max', type=int, default=130000)

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt_files = sorted(glob.glob(os.path.join(args.ckpt_dir, '[0-9]'*6 + '.model')))
    ckpt_files = [f for f in ckpt_files if args.ckpt_min <= int(os.path.basename(f)[:6]) <= args.ckpt_max]

    for ckpt_path in ckpt_files:
        step_str = os.path.basename(ckpt_path)[:6]
        ckpt_dir = os.path.join(args.save_dir, step_str)
        os.makedirs(ckpt_dir, exist_ok=True)

        generator = StyledGenerator(512).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            generator.load_state_dict(ckpt)
        except RuntimeError:
            generator.load_state_dict(ckpt['g_running'])
        generator.eval()

        for label in args.labels:
            label_dir = os.path.join(ckpt_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            count = 0

            for b in range(0, args.n_per_label, args.batch_size):
                current_bs = min(args.batch_size, args.n_per_label - b)
                z = torch.randn(current_bs, 512, device=device)
                label_tensor = torch.full((current_bs,), label, dtype=torch.float32, device=device)
                with torch.no_grad():
                    imgs = generator(z, label_tensor, step=args.step, alpha=1)
                for img in imgs:
                    utils.save_image(img, os.path.join(label_dir, f"{count}.png"), normalize=True, value_range=(-1, 1))
                    count += 1
            print(f"Step {step_str}, Label {label}: Saved {count} images in {label_dir}")

if __name__ == '__main__':
    main()



