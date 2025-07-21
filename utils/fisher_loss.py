import torch
from utils.bessel_function import bessel0, bessel1


def torch_integral(f, v, from_x, to_x, N):
    with torch.no_grad():
        rangee = torch.arange(N, dtype=v.dtype, device=v.device)
        x = (rangee * ((to_x - from_x) / (N - 1)) + from_x).view(1, N)
        weights = torch.empty((1, N), dtype=v.dtype, device=v.device).fill_(1)
        weights[0, 0] = 1 / 2
        weights[0, -1] = 1 / 2
        y = f(x, v)
        return torch.sum(y * weights, dim=1) * (to_x - from_x) / (N - 1)


def integrand_aF(x, s):
    sk = s[:, 2]
    sj = s[:, 1]
    si = s[:, 0]

    min_ij = torch.min(s[:, :2], dim=-1).values
    f1 = (si - sj) / 2
    f2 = (si + sj) / 2

    a1 = f1.view(-1, 1) * (1 - x).view(1, -1)
    a2 = f2.view(-1, 1) * (1 + x).view(1, -1)
    a3 = (sk + min_ij).view(-1, 1) * (x - 1).view(1, -1)

    i1 = bessel0(a1)
    i2 = bessel0(a2)
    i3 = torch.exp(a3)
    ret = i1 * i2 * i3
    return ret


def integrand_aF_diff3(x, s):
    sk = s[:, 2]
    sj = s[:, 1]
    si = s[:, 0]
    min_ij = torch.min(s[:, :2], dim=-1).values

    f1 = (si - sj) / 2
    f2 = (si + sj) / 2

    a1 = f1.view(-1, 1) * (1 - x).view(1, -1)
    a2 = f2.view(-1, 1) * (1 + x).view(1, -1)

    i01 = bessel0(a1)
    i02 = bessel0(a2)
    a3 = (min_ij + sk).view(-1, 1) * (x - 1).view(1, -1)

    i3 = torch.exp(a3)
    i4 = (x - 1).view(1, -1)
    return i01 * i02 * i3 * i4


def integrand_aF_diff1(x, s):
    sk = s[:, 2]
    sj = s[:, 1]
    si = s[:, 0]
    min_ij, min_indices = torch.min(s[:, :2], dim=-1)
    f1 = (si - sj) / 2
    f2 = (si + sj) / 2

    a1 = f1.view(-1, 1) * (1 - x).view(1, -1)
    a2 = f2.view(-1, 1) * (1 + x).view(1, -1)
    a3 = (min_ij + sk).view(-1, 1) * (x - 1).view(1, -1)

    i01 = bessel1(a1) - torch.sign(a1) * bessel0(a1)
    i02 = bessel0(a2)
    i3 = torch.exp(a3)
    i04 = (1 - x).view(1, -1) / 2

    i0 = i01 * i02 * i3 * i04

    i11 = bessel0(a1)
    i12 = bessel1(a2) - torch.sign(a2) * bessel0(a2)
    i14 = (1 + x).view(1, -1) / 2

    i1 = i11 * i12 * i3 * i14

    i21 = i11
    i22 = i02

    i4 = (x - 1).view(1, -1)
    i2 = i21 * i22 * i3 * i4
    i2[min_indices == 1] = 0

    return i0 + i1 + i2


def integrand_aF_diff2(x, s):
    sk = s[:, 2]
    sj = s[:, 1]
    si = s[:, 0]
    min_ij, min_indices = torch.min(s[:, :2], dim=-1)
    f1 = (si - sj) / 2
    f2 = (si + sj) / 2

    a1 = f1.view(-1, 1) * (1 - x).view(1, -1)
    a2 = f2.view(-1, 1) * (1 + x).view(1, -1)
    a3 = (min_ij + sk).view(-1, 1) * (x - 1).view(1, -1)

    i01 = bessel1(a1) - torch.sign(a1) * bessel0(a1)
    i02 = bessel0(a2)
    i3 = torch.exp(a3)
    i04 = (1 - x).view(1, -1) / 2

    i0 = i01 * i02 * i3 * i04  # * i4

    i11 = bessel0(a1)
    i12 = bessel1(a2) - torch.sign(a2) * bessel0(a2)
    i14 = (1 + x).view(1, -1) / 2

    i1 = i11 * i12 * i3 * i14  # * i4
    i21 = i11
    i22 = i02

    i4 = (x - 1).view(1, -1)

    i2 = i21 * i22 * i3 * i4
    i2[min_indices == 0] = 0

    return -i0 + i1 + i2


class class_logC_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        N = 128
        shape = input.shape
        input_v = input.view(-1, 3)
        factor = 1 / 2 * torch_integral(integrand_aF, input_v, -1, 1, N)
        ctx.save_for_backward(input, factor)
        log_factor = torch.log(factor)
        return log_factor.view(shape[:-1])

    @staticmethod
    def backward(ctx, grad):
        S, factor = ctx.saved_tensors
        S = S.view(-1, 3)
        N = 128
        ret = torch.empty((S.shape[0], 3), dtype=S.dtype, device=S.device)
        for i in range(3):
            if i == 0:
                ret[:, i] = 1 / 2 * torch_integral(integrand_aF_diff1, S, -1, 1, N)
            elif i == 1:
                ret[:, i] = 1 / 2 * torch_integral(integrand_aF_diff2, S, -1, 1, N)
            else:
                ret[:, i] = 1 / 2 * torch_integral(integrand_aF_diff3, S, -1, 1, N)
        ret /= factor.view(-1, 1)
        ret *= grad.view(-1, 1)
        return ret.view((*grad.shape, 3))


logC_F = class_logC_F.apply


def KL_Fisher(A, R, overreg=1.05):
    U, S, Vh = torch.linalg.svd(A)
    S_sign = S.clone()
    with torch.no_grad():
        rotation_candidate = torch.matmul(U, Vh.transpose(1, 2))
        s3sign = torch.det(rotation_candidate)

    S_sign[:, 2] *= s3sign
    log_normalizer = logC_F(S_sign)

    log_exponent = -torch.matmul(A.view(-1, 1, 9), R.view(-1, 9, 1)).view(-1)
    log_suppress = torch.sum(S_sign, dim=-1)

    loss_values = log_exponent + overreg * (log_normalizer + log_suppress)

    return loss_values


def vmf_loss(net_out, R, overreg=1.05):
    A = net_out.reshape(-1, 3, 3)
    loss_v = KL_Fisher(A, R, overreg=overreg)
    if loss_v is None:
        Rest = torch.unsqueeze(torch.eye(3, 3, device=R.device, dtype=R.dtype), 0)
        Rest = torch.repeat_interleave(Rest, R.shape[0], 0)
        return None

    return loss_v.mean()
