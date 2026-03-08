import torch


def _skew(v):
    zero = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack(
        [
            torch.stack([zero, -v[2], v[1]]),
            torch.stack([v[2], zero, -v[0]]),
            torch.stack([-v[1], v[0], zero]),
        ]
    )


def so3_exp_map(omega):
    theta = torch.linalg.norm(omega)
    omega_hat = _skew(omega)
    omega_hat_sq = omega_hat @ omega_hat
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device)

    eps = 1e-8
    if theta < eps:
        return identity + omega_hat + 0.5 * omega_hat_sq

    theta_sq = theta * theta
    a = torch.sin(theta) / theta
    b = (1.0 - torch.cos(theta)) / theta_sq
    return identity + a * omega_hat + b * omega_hat_sq


def se3_exp_map(xi):
    omega = xi[:3]
    upsilon = xi[3:]
    theta = torch.linalg.norm(omega)

    omega_hat = _skew(omega)
    omega_hat_sq = omega_hat @ omega_hat
    identity = torch.eye(3, dtype=xi.dtype, device=xi.device)

    eps = 1e-8
    if theta < eps:
        rotation = identity + omega_hat + 0.5 * omega_hat_sq
        v_matrix = identity + 0.5 * omega_hat + (1.0 / 6.0) * omega_hat_sq
    else:
        theta_sq = theta * theta
        theta_cu = theta_sq * theta
        a = torch.sin(theta) / theta
        b = (1.0 - torch.cos(theta)) / theta_sq
        c = (theta - torch.sin(theta)) / theta_cu
        rotation = identity + a * omega_hat + b * omega_hat_sq
        v_matrix = identity + b * omega_hat + c * omega_hat_sq

    translation = v_matrix @ upsilon

    transform = torch.eye(4, dtype=xi.dtype, device=xi.device)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def apply_se3_delta(c2w, pose_delta):
    delta_transform = se3_exp_map(pose_delta)
    return c2w @ delta_transform
