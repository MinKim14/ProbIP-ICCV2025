import torch

bessel0_a = [
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.360768e-1,
    0.45813e-2,
][::-1]
bessel0_b = [
    0.39894228,
    0.1328592e-1,
    0.225319e-2,
    -0.157565e-2,
    0.916281e-2,
    -0.2057706e-1,
    0.2635537e-1,
    -0.1647633e-1,
    0.392377e-2,
][::-1]

bessel1_a = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.2658733e-1, 0.301532e-2, 0.32411e-3][
    ::-1
]
bessel1_b = [
    0.39894228,
    -0.3988024e-1,
    -0.362018e-2,
    0.163801e-2,
    -0.1031555e-1,
    0.2282967e-1,
    -0.2895312e-1,
    0.1787654e-1,
    -0.420059e-2,
][::-1]


def _horner(arr, x):
    z = torch.empty(x.shape, dtype=x.dtype, device=x.device).fill_(arr[0])
    for i in range(1, len(arr)):
        z.mul_(x).add_(arr[i])
    return z


def bessel0(x):  # always supressed by exp(x)
    abs_x = torch.abs(x)
    mask = abs_x <= 3.75
    e1 = _horner(bessel0_a, (abs_x / 3.75) ** 2) / torch.exp(abs_x)
    e2 = _horner(bessel0_b, 3.75 / abs_x) / torch.sqrt(abs_x)
    e2[mask] = e1[mask]
    return e2


def bessel1(x):
    abs_x = torch.abs(x)
    mask = abs_x <= 3.75
    e1 = x * _horner(bessel1_a, (abs_x / 3.75) ** 2) / torch.exp(abs_x)
    sign_x = torch.sign(x)
    e2 = sign_x * _horner(bessel1_b, 3.75 / abs_x) / torch.sqrt(abs_x)
    e2[mask] = e1[mask]
    return e2
