"""Pure-PyTorch implementations of the pytorch3d.transforms functions MACE imports.

Formulas follow the standard quaternion conventions used by pytorch3d
(w, x, y, z ordering, Hamilton product, right-handed rotations).
"""
import math

import torch


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError('Invalid rotation matrix shape.')
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    w = 0.5 * torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0))
    x = 0.5 * torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0)) * torch.sign(m21 - m12)
    y = 0.5 * torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0)) * torch.sign(m02 - m20)
    z = 0.5 * torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0)) * torch.sign(m10 - m01)
    return torch.stack((w, x, y, z), -1)


def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half = angles * 0.5
    small = angles < 1e-6
    sin = torch.where(small, torch.full_like(half, 0.5), torch.sin(half) / angles.clamp_min(1e-12))
    xyz = axis_angle * sin
    w = torch.where(small, torch.ones_like(half), torch.cos(half))
    return torch.cat((w, xyz), dim=-1)


def quaternion_to_axis_angle(quaternions):
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    small = angles < 1e-6
    sin_half = torch.where(small, torch.full_like(half_angles, 0.5), norms / angles.clamp_min(1e-12))
    return quaternions[..., 1:] / sin_half.clamp_min(1e-12)


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = torch.unbind(quaternion0, -1)
    w1, x1, y1, z1 = torch.unbind(quaternion1, -1)
    return torch.stack((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0), dim=-1)


def quaternion_apply(quaternion, point):
    if point.size(-1) != 3:
        raise ValueError('Points must have last dimension 3')
    w, x, y, z = torch.unbind(quaternion, -1)
    two_s = 2.0
    px, py, pz = torch.unbind(point, -1)
    return torch.stack((
        px * (1 - two_s * (y * y + z * z)) + py * (two_s * (x * y - w * z)) + pz * (two_s * (x * z + w * y)),
        px * (two_s * (x * y + w * z)) + py * (1 - two_s * (x * x + z * z)) + pz * (two_s * (y * z - w * x)),
        px * (two_s * (x * z - w * y)) + py * (two_s * (y * z + w * x)) + pz * (1 - two_s * (x * x + y * y))), -1)


def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class RotateAxisAngle:
    """Rotation about a fixed axis, matching pytorch3d.transforms.RotateAxisAngle."""

    def __init__(self, angle, axis='Z', degrees=True, device=None):
        if degrees:
            angle = torch.tensor(angle) * math.pi / 180.0 if not torch.is_tensor(angle) else angle * math.pi / 180.0
        self.angle = angle
        self.axis = axis.upper()
        self.device = device

    def get_matrix(self):
        if not torch.is_tensor(self.angle):
            self.angle = torch.tensor(self.angle, device=self.device)
        device = self.angle.device
        zero = torch.zeros_like(self.angle)
        one = torch.ones_like(self.angle)
        c = torch.cos(self.angle)
        s = torch.sin(self.angle)
        if self.axis == 'X':
            rows = ([one, zero, zero], [zero, c, -s], [zero, s, c])
        elif self.axis == 'Y':
            rows = ([c, zero, s], [zero, one, zero], [-s, zero, c])
        else:
            rows = ([c, -s, zero], [s, c, zero], [zero, zero, one])
        return torch.stack([torch.stack(row, -1) for row in rows], -2)

    def transform_points(self, points):
        matrix = self.get_matrix().to(device=points.device, dtype=points.dtype)
        return points @ matrix.transpose(-1, -2)

    def __add__(self, other):
        return _Compose([self, other])

    def __radd__(self, other):
        return _Compose([other, self])


class _Compose:
    def __init__(self, items):
        self.items = items

    def get_matrix(self):
        result = None
        for item in self.items:
            matrix = item.get_matrix() if hasattr(item, 'get_matrix') else torch.as_tensor(item)
            result = matrix if result is None else result @ matrix
        return result

    def transform_points(self, points):
        matrix = self.get_matrix().to(device=points.device, dtype=points.dtype)
        return points @ matrix.transpose(-1, -2)

    def __add__(self, other):
        return _Compose(self.items + [other])

    def __radd__(self, other):
        return _Compose([other] + self.items)
