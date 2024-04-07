import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor


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
    tan_fovx = 0.5 * img_width / fx
    tan_fovy = 0.5 * img_height / fy
    p_view, is_close = clip_near_plane(means3d, viewmat, clip_thresh)
    cov3d = scale_rot_to_cov3d(scales, glob_scale, quats)
    cov2d, compensation = project_cov3d_ewa(means3d, cov3d, viewmat, fx, fy, tan_fovx, tan_fovy)
    conic, radius, det_valid = compute_cov2d_bounds(cov2d)
    xys = project_pix(fx, fy, cx, cy, p_view)

    depths = p_view[..., 2]
    radii = radius.to(torch.int32)
    mask = (~is_close) & det_valid

    radii = torch.where(~mask, 0, radii)
    conic = torch.where(~mask[..., None], 0, conic)
    xys = torch.where(~mask[..., None], 0, xys)
    cov3d = torch.where(~mask[..., None, None], 0, cov3d)
    cov2d = torch.where(~mask[..., None, None], 0, cov2d)
    compensation = torch.where(~mask, 0, compensation)
    depths = torch.where(~mask, 0, depths)

    i, j = torch.triu_indices(3, 3)
    cov3d_triu = cov3d[..., i, j]
    return (
        xys,
        depths,
        radii,
        conic,
        compensation,
        cov3d_triu,
    )


def clip_near_plane(
    p: Float[Tensor, "n 3"], viewmat: Float[Tensor, " 4 4"], clip_thresh: float = 0.01
) -> tuple[Float[Tensor, " n 3"], Bool[Tensor, " n"]]:
    R = viewmat[:3, :3]
    T = viewmat[:3, 3]
    p_view = torch.einsum("ij,nj->ni", R, p) + T[None]
    return p_view, p_view[..., 2] < clip_thresh


def scale_rot_to_cov3d(
    scale: Float[Tensor, "*n 3"], glob_scale: float, quat: Float[Tensor, "*n 4"]
) -> Float[Tensor, "*n 3 3"]:
    R = quat_to_rotmat(quat)  # (..., 3, 3)
    M = R * glob_scale * scale[..., None, :]  # (..., 3, 3)
    return M @ M.transpose(-1, -2)  # (..., 3, 3)


def quat_to_rotmat(quat: Float[Tensor, "*n 4"]) -> Float[Tensor, "*n 3 3"]:
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def project_cov3d_ewa(
    mean3d: Float[Tensor, "*batch 3"],
    cov3d: Float[Tensor, "*batch 3 3"],
    viewmat: Float[Tensor, "4 4"],
    fx: float,
    fy: float,
    tan_fovx: float,
    tan_fovy: float,
) -> tuple[Float[Tensor, "*batch 2 2"], Float[Tensor, " *batch"]]:
    assert mean3d.shape[-1] == 3, mean3d.shape
    assert cov3d.shape[-2:] == (3, 3), cov3d.shape
    assert viewmat.shape[-2:] == (4, 4), viewmat.shape
    W = viewmat[..., :3, :3]  # (..., 3, 3)
    p = viewmat[..., :3, 3]  # (..., 3)
    t = torch.einsum("...ij,...j->...i", W, mean3d) + p  # (..., 3)

    rz = 1.0 / t[..., 2]  # (...,)
    rz2 = rz**2  # (...,)

    lim_x = 1.3 * torch.tensor([tan_fovx], device=mean3d.device)
    lim_y = 1.3 * torch.tensor([tan_fovy], device=mean3d.device)
    x_clamp = t[..., 2] * torch.clamp(t[..., 0] * rz, min=-lim_x, max=lim_x)
    y_clamp = t[..., 2] * torch.clamp(t[..., 1] * rz, min=-lim_y, max=lim_y)
    t = torch.stack([x_clamp, y_clamp, t[..., 2].clone()], dim=-1)

    O = torch.zeros_like(rz)
    J = torch.stack(
        [fx * rz, O, -fx * t[..., 0] * rz2, O, fy * rz, -fy * t[..., 1] * rz2],
        dim=-1,
    ).reshape(*rz.shape, 2, 3)
    T = torch.matmul(J, W)  # (..., 2, 3)
    cov2d = torch.einsum("...ij,...jk,...kl->...il", T, cov3d, T.transpose(-1, -2))
    det_orig = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 0, 1]
    cov2d[..., 0, 0] = cov2d[..., 0, 0] + 0.3
    cov2d[..., 1, 1] = cov2d[..., 1, 1] + 0.3
    det_blur = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 0, 1]
    compensation = torch.sqrt(torch.clamp(det_orig / det_blur, min=0))
    return cov2d[..., :2, :2], compensation


def compute_cov2d_bounds(
    cov2d_mat: Float[Tensor, "*batch 2 2"],
) -> tuple[Float[Tensor, "*batch 3"], Float[Tensor, " *batch"], Bool[Tensor, " *batch"]]:
    det_all = cov2d_mat[..., 0, 0] * cov2d_mat[..., 1, 1] - cov2d_mat[..., 0, 1] ** 2
    valid = det_all != 0
    # det = torch.clamp(det, min=eps)
    det = det_all[valid]
    cov2d = cov2d_mat[valid]
    conic = torch.stack(
        [
            cov2d[..., 1, 1] / det,
            -cov2d[..., 0, 1] / det,
            cov2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # (..., 3)
    b = (cov2d[..., 0, 0] + cov2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=0.1))  # (...,)
    v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.1))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)
    radius_all = torch.zeros(*cov2d_mat.shape[:-2], device=cov2d_mat.device)
    conic_all = torch.zeros(*cov2d_mat.shape[:-2], 3, device=cov2d_mat.device)
    radius_all[valid] = radius
    conic_all[valid] = conic
    return conic_all, radius_all, valid


def project_pix(
    fx: float, fy: float, cx: float, cy: float, p_view: Float[Tensor, " *batch 3"], eps: float = 1e-6
) -> Float[Tensor, " *batch 2"]:
    rw = 1.0 / (p_view[..., 2] + eps)
    p_proj = (p_view[..., 0] * rw, p_view[..., 1] * rw)
    u, v = (p_proj[0] * fx + cx, p_proj[1] * fy + cy)
    return torch.stack([u, v], dim=-1)
