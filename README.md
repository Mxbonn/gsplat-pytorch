# gsplat-pytorch

gsplat-pytorch is a PyTorch implementation of rasterize and project functions used in the [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) paper.

The design is inspired by the [gsplat](https://github.com/nerfstudio-project/gsplat) library for from the Nerfstudio-team.

This implementation does not use CUDA and is therefore runnable on all PyTorch supported devices.
The CUDA implementation can be found in the [gsplat](https://github.com/nerfstudio-project/gsplat) library and should be used for any performance critical applications, this is implementation is mainly intended for research and educational purposes.

## API
The functions are similar those in [gsplat library](https://docs.gsplat.studio/apis/proj.html) with the main difference being the removal of the `num_tiles_hit` and `return_alpha` parameters.
```python
def project_gaussians(
    means3d: Float[Tensor, "*batch 3"],
    scales: Float[Tensor, "*batch 3"],
    glob_scale: float,
    quats: Float[Tensor, "*batch 4"],
    viewmat: Float[Tensor, "4 4"],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    clip_thresh: float = 0.01,
) -> tuple[
    Float[Tensor, " *batch 2"],
    Float[Tensor, " *batch"],
    Float[Tensor, " *batch"],
    Float[Tensor, " *batch 3"],
    Float[Tensor, " *batch"],
    Float[Tensor, " *batch 3 3"],
]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       clip_thresh (float): minimum z depth threshold.

    Returns:
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        conics (Tensor): conic parameters for 2D gaussian.
        compensation (Tensor): the density compensation for blurring 2D kernel
        cov3d (Tensor): 3D covariances.
    """

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
```

`rasterize_gaussians` implements a tile based approach, while `rasterize_gaussians_simple_loop` implements a simple loop based approach, and `rasterize_gaussian_vectorized` implements a fully vectorized approach without loops or tiling.

## Installation
```bash
pip install -e .
```
## Example

Fit a 2D image with 3D Gaussians.

```bash
pip install -r examples/requirements.txt
python examples/simple_trainer.py
```