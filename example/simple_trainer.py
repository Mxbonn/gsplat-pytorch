import math
import os
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor, optim
from tqdm import tqdm

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussian


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""

        self.means = 2 * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        self.rgbs = torch.rand(self.num_points, 3, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(3, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
    ):
        optimizer = optim.Adam([self.rgbs, self.means, self.scales, self.opacities, self.quats], lr)
        mse_loss = torch.nn.MSELoss()
        frames = []
        for iter in (pbar := tqdm(range(iterations))):
            (
                xys,
                depths,
                radii,
                conics,
                compensation,
                cov3d,
            ) = project_gaussians(
                self.means,
                self.scales,
                1,
                self.quats / torch.linalg.norm(self.quats, dim=1, keepdim=True),
                self.viewmat,
                self.focal,
                self.focal,
                self.W / 2,
                self.H / 2,
                self.H,
                self.W,
            )
            out_img = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
                8,
                self.background,
            )
            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
    device: str = "cpu",
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points, device=device)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--save-imgs", action=BooleanOptionalAction)
    parser.add_argument("--img-path", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(
        height=args.height,
        width=args.width,
        num_points=args.num_points,
        save_imgs=args.save_imgs,
        img_path=args.img_path,
        iterations=args.iterations,
        lr=args.lr,
    )
