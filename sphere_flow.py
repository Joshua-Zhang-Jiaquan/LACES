"""Geodesic (Riemannian) flow matching on the hypersphere S^{d-1}(R).

NOT Euclidean. Latents are constrained to a fixed-radius sphere of radius R=sqrt(d);
interpolation is the geodesic slerp along great circles, the velocity target is the
tangent vector along that geodesic, and the ODE sampler steps via the exponential map
so every iterate stays exactly on the sphere. Follows Riemannian Flow Matching
(Chen & Lipman, arXiv:2302.03660) and the sphere-latent recipe of
"Aligning Latent Geometry for Spherical Flow Matching" (arXiv:2605.15193).

All functions operate on the last dimension; batch/other dims are broadcast.
"""

import torch

_EPS = 1e-6


def project_to_sphere(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Radial projection of any vector onto the sphere of the given radius."""
    n = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return x / n * radius


def sample_sphere_uniform(shape, radius: float, device, dtype) -> torch.Tensor:
    """Uniform sample on S^{d-1}(R): normalize an isotropic Gaussian."""
    g = torch.randn(shape, device=device, dtype=dtype)
    return project_to_sphere(g, radius)


def proju(x: torch.Tensor, v: torch.Tensor, radius: float) -> torch.Tensor:
    """Project ambient vector v onto the tangent space at x (radius-R sphere)."""
    coef = (v * x).sum(dim=-1, keepdim=True) / (radius * radius)
    return v - coef * x


def _angle(x0: torch.Tensor, x1: torch.Tensor, radius: float) -> torch.Tensor:
    cos = (x0 * x1).sum(dim=-1, keepdim=True) / (radius * radius)
    return cos.clamp(-1.0 + _EPS, 1.0 - _EPS).acos()


def slerp(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, radius: float) -> torch.Tensor:
    """Geodesic interpolation on the sphere; t broadcastable to x0's leading dims."""
    omega = _angle(x0, x1, radius)
    so = omega.sin().clamp_min(_EPS)
    a = ((1.0 - t) * omega).sin() / so
    b = (t * omega).sin() / so
    return a * x0 + b * x1


def slerp_velocity(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, radius: float) -> torch.Tensor:
    """d/dt slerp: the tangent velocity along the geodesic (constant speed = R*omega)."""
    omega = _angle(x0, x1, radius)
    so = omega.sin().clamp_min(_EPS)
    a = -omega * ((1.0 - t) * omega).cos() / so
    b = omega * (t * omega).cos() / so
    return a * x0 + b * x1


def expmap(x: torch.Tensor, u: torch.Tensor, radius: float) -> torch.Tensor:
    """Exponential map: move from x along tangent u, staying on the sphere.

    On S^{d-1}(R), exp_x(u) = cos(|u|/R)*x + R*sin(|u|/R)*u/|u|. Falls back to a
    radial retraction of x+u when |u| is tiny.
    """
    un = u.norm(dim=-1, keepdim=True)
    theta = un / radius
    small = un < _EPS
    exp = theta.cos() * x + radius * theta.sin() * u / un.clamp_min(_EPS)
    retr = project_to_sphere(x + u, radius)
    return torch.where(small, retr, exp)
