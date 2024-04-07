"""Python bindings for custom Cuda functions"""

import torch
from jaxtyping import Float
from torch import Tensor


def rasterize_gaussians(
    xys: Float[Tensor, " *batch 2"],
    depths: Float[Tensor, " *batch 1"],
    radii: Float[Tensor, " *batch 1"],
    conics: Float[Tensor, " *batch 3"],
    colors: Float[Tensor, " *batch channels"],
    opacity: Float[Tensor, " *batch"],
    img_height: int,
    img_width: int,
    block_width: int,
    background: Float[Tensor, " channels"],
) -> Float[Tensor, " img_height img_width channels"]:
    """Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        block_width (int): width of the tiling block for rasterization.
        background (Tensor): background color (rgb).

    Returns:
        out_img (Tensor): N-dimensional rendered output image.
    """
    depth_sorted_idxs = torch.argsort(depths)
    depth_sorted_xys = xys[depth_sorted_idxs]
    depth_sorted_radii = radii[depth_sorted_idxs]
    depth_sorted_conics = conics[depth_sorted_idxs]
    depth_sorted_colors = colors[depth_sorted_idxs]
    depth_sorted_opacity = opacity[depth_sorted_idxs]

    channels = colors.shape[1]
    out_img = torch.zeros((img_height, img_width, channels), dtype=torch.float32, device=depth_sorted_xys.device)

    gaussian_2d_bounds = torch.stack(
        [depth_sorted_xys - depth_sorted_radii[..., None], depth_sorted_xys + depth_sorted_radii[..., None]], dim=-1
    )

    for i in range(0, img_width, block_width):
        for j in range(0, img_height, block_width):
            tile_x_min, tile_x_max = i, i + block_width
            tile_y_min, tile_y_max = j, j + block_width
            gaussian_x_min, gaussian_x_max = gaussian_2d_bounds[:, 0, 0], gaussian_2d_bounds[:, 0, 1]
            gaussian_y_min, gaussian_y_max = gaussian_2d_bounds[:, 1, 0], gaussian_2d_bounds[:, 1, 1]
            x_intersect = torch.logical_and(gaussian_x_min <= tile_x_max, gaussian_x_max >= tile_x_min)
            y_intersect = torch.logical_and(gaussian_y_min <= tile_y_max, gaussian_y_max >= tile_y_min)
            intersect = torch.logical_and(x_intersect, y_intersect)

            if not torch.any(intersect):
                continue
            y_coords, x_coords = torch.meshgrid(
                torch.arange(j, j + block_width), torch.arange(i, i + block_width), indexing="ij"
            )
            pixel_coords = torch.stack([x_coords, y_coords], dim=-1).to(depth_sorted_xys.device)
            xys_tile = depth_sorted_xys[intersect]
            conics_tile = depth_sorted_conics[intersect]
            colors_tile = depth_sorted_colors[intersect]
            opacity_tile = depth_sorted_opacity[intersect]

            # Compute delta for all pixels and gaussians
            delta = xys_tile[:, None, None, :] - pixel_coords[None, ...]
            # Compute sigma for all pixels and gaussians
            sigma = (
                0.5
                * (
                    conics_tile[:, None, None, 0] * delta[..., 0] ** 2
                    + conics_tile[:, None, None, 2] * delta[..., 1] ** 2
                )
                + conics_tile[:, None, None, 1] * delta[..., 0] * delta[..., 1]
            )

            sigma = torch.clamp(sigma, min=0)
            # Compute alpha for all pixels and gaussians
            alpha = torch.clamp(opacity_tile[..., None, None] * torch.exp(-sigma), max=1)

            # Compute visibility for all pixels and gaussians
            next_T = torch.cumprod(1 - alpha, dim=0)
            T = torch.cat([torch.ones_like(next_T[:1]), next_T[:-1]], dim=0)
            vis = alpha * T

            # Compute the output image
            out_img[j : j + block_width, i : i + block_width] = torch.sum(
                vis[..., None] * colors_tile[:, None, None, :], dim=0
            )
            out_img[j : j + block_width, i : i + block_width] += next_T[-1].unsqueeze(-1).clone() * background

    return out_img


def rasterize_gaussians_simple_loop(
    xys: Float[Tensor, " *batch 2"],
    depths: Float[Tensor, " *batch 1"],
    radii: Float[Tensor, " *batch 1"],  # NOT USED IN THIS VERSION
    conics: Float[Tensor, " *batch 3"],
    colors: Float[Tensor, " *batch channels"],
    opacity: Float[Tensor, " *batch"],
    img_height: int,
    img_width: int,
    block_width: int,  # NOT USED IN THIS VERSION
    background: Float[Tensor, " channels"],
) -> Float[Tensor, " img_height img_width channels"]:
    """Rasterizes 2D gaussians and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians # NOT USED IN THIS VERSION
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        block_width (int): width of the tiling block for rasterization. # NOT USED IN THIS VERSION
        background (Tensor): background color (rgb).

    Returns:
        out_img (Tensor): N-dimensional rendered output image.
    """
    depth_sorted_idxs = torch.argsort(depths)
    depth_sorted_xys = xys[depth_sorted_idxs]
    depth_sorted_conics = conics[depth_sorted_idxs]
    depth_sorted_colors = colors[depth_sorted_idxs]
    depth_sorted_opacity = opacity[depth_sorted_idxs]

    channels = colors.shape[1]
    out_img = torch.zeros((img_height, img_width, channels), dtype=torch.float32, device=depth_sorted_xys.device)

    for i in range(img_height):
        for j in range(img_width):
            T = 1.0
            for center, conic, color, opac in zip(
                depth_sorted_xys, depth_sorted_conics, depth_sorted_colors, depth_sorted_opacity
            ):
                delta = center - torch.tensor([j, i], dtype=torch.float32, device=xys.device)

                sigma = (
                    0.5 * (conic[0] * delta[0] * delta[0] + conic[2] * delta[1] * delta[1])
                    + conic[1] * delta[0] * delta[1]
                )

                if sigma < 0:
                    continue

                alpha = torch.clamp(opac * torch.exp(-sigma), max=1)

                if alpha < 1 / 255:
                    continue

                next_T = T * (1 - alpha)

                if next_T <= 1e-4:
                    break

                vis = alpha * T

                out_img[i, j] += vis * color
                T = next_T

            out_img[i, j] += T * background

    return out_img


def rasterize_gaussians_vectorized(
    xys: Float[Tensor, " *batch 2"],
    depths: Float[Tensor, " *batch 1"],
    radii: Float[Tensor, " *batch 1"],  # NOT USED IN THIS VERSION
    conics: Float[Tensor, " *batch 3"],
    colors: Float[Tensor, " *batch channels"],
    opacity: Float[Tensor, " *batch"],
    img_height: int,
    img_width: int,
    block_width: int,  # NOT USED IN THIS VERSION
    background: Float[Tensor, " channels"],
) -> Float[Tensor, " img_height img_width channels"]:
    """Rasterizes 2D gaussians and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians # NOT USED IN THIS VERSION
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        block_width (int): width of the tiling block for rasterization. # NOT USED IN THIS VERSION
        background (Tensor): background color (rgb).

    Returns:
        out_img (Tensor): N-dimensional rendered output image.
    """
    depth_sorted_idxs = torch.argsort(depths)
    depth_sorted_xys = xys[depth_sorted_idxs]
    depth_sorted_conics = conics[depth_sorted_idxs]
    depth_sorted_colors = colors[depth_sorted_idxs]
    depth_sorted_opacity = opacity[depth_sorted_idxs]

    channels = colors.shape[1]
    out_img = torch.zeros((img_height, img_width, channels), dtype=torch.float32, device=depth_sorted_xys.device)
    # Create a grid of pixel coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(img_height), torch.arange(img_width), indexing="ij")
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1).to(xys.device)

    # Compute delta for all pixels and gaussians
    delta = depth_sorted_xys[:, None, None, :] - pixel_coords[None, ...]
    # Compute sigma for all pixels and gaussians
    sigma = (
        0.5
        * (
            depth_sorted_conics[:, None, None, 0] * delta[..., 0] ** 2
            + depth_sorted_conics[:, None, None, 2] * delta[..., 1] ** 2
        )
        + depth_sorted_conics[:, None, None, 1] * delta[..., 0] * delta[..., 1]
    )

    sigma = torch.clamp(sigma, min=0)
    # Compute alpha for all pixels and gaussians
    alpha = torch.clamp(depth_sorted_opacity[..., None, None] * torch.exp(-sigma), max=1)

    # Compute visibility for all pixels and gaussians
    next_T = torch.cumprod(1 - alpha, dim=0)
    T = torch.cat([torch.ones_like(next_T[:1]), next_T[:-1]], dim=0)
    vis = alpha * T

    # Compute the output image
    out_img = torch.sum(
        vis.unsqueeze(-1) * depth_sorted_colors[:, None, None, :],
        dim=0,
    )
    out_img += next_T[-1, ..., None] * background

    return out_img
