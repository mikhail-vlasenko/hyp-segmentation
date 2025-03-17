import time

import geoopt as gt
import torch
from geoopt.manifolds.stereographic.math import artan_k, _mobius_add
import torch.nn.functional as F

ball = gt.PoincareBall(c=1.)


def usual_dist(_reps, _prototypes):
    return ball.dist(_reps, _prototypes)

def internal_dist(_reps, _prototypes):
    dim = -1
    keepdim = False
    k = ball.k
    mobius_norm2 = _mobius_add(-_reps, _prototypes, k, dim=dim).norm(dim=dim, p=2, keepdim=keepdim)
    return 2.0 * artan_k(
        mobius_norm2, k
    )

def pasted_norm(inputs, P_mlr, c):
    def torch_sqnorm(u, keepdim=True, dim=-1):
        # performs sq norm over last dim
        return torch.sum(u * u, dim=dim, keepdim=keepdim)

    def cross_correlate_torch(inputs, filters):
        return F.conv2d(inputs, filters, stride=1, padding=(filters.shape[-1] - 1) // 2)

    EPS = 1e-15
    xx = torch_sqnorm(inputs)
    pp = torch_sqnorm(-P_mlr, keepdim=False, dim=1)
    assert not torch.isnan(pp).any()
    # 1x1 conv.
    # | -p * x |^2 p has shape ncls, D, need in-out shape for filter: D,ncls
    P_kernel = torch.transpose(-P_mlr, -1, 0)[None, None, :, :]

    inputs = inputs.permute((0, 3, 1, 2))
    P_kernel = P_kernel.permute((3, 2, 0, 1))

    px = cross_correlate_torch(inputs, P_kernel)
    px = px.permute((0, 2, 3, 1))

    # c^2 * | X|^2 * |-P|^2
    sqsq = torch.mul(c * xx, c * pp[None, None, None, :])  # sh B,H,W,ch

    # rewrite mob add as alpha * p + beta * x
    # where alpha = A/D
    A = 1 + torch.add(2 * c * px, c * xx)  # sh B,H,W,ch

    assert not torch.isnan(A).any()

    B = 1 - c * pp  # sh ch ## beta = B/D
    assert not torch.isnan(B).any()

    D = 1 + torch.add(2 * c * px, sqsq)  # sh B,H,W,ch
    D = torch.maximum(D, torch.tensor(EPS))
    assert not torch.isnan(D).any()

    # calculate mobadd norm indepently from mob add
    # if mob_add = alpha * p + beta * x, then
    #  |mob_add|^2 = theta**2 * |p|^2 + gamma^2 * |x|^2 + 2*theta*gamma*|px|
    # theta = A/D, gamma = B/D
    alpha = A / D  # B,H,W,ch
    assert not torch.isnan(alpha).any()
    beta = B[None, None, None, :] / D  # B,H,W,ch
    assert not torch.isnan(beta).any()

    # calculate mobius addition norm independently
    mobaddnorm = (
            (alpha ** 2 * pp[None, None, None, :])
            + (beta ** 2 * xx)
            + (2 * alpha * beta * px)
    )
    return mobaddnorm

def geoopt_norm(p, z):
    dim = -1
    keepdim = False
    c = ball.k
    return _mobius_add(-p, z, c, dim=dim).norm(dim=dim, p=2, keepdim=keepdim)

def my_norm(p, z):
    c = ball.k
    p = -p
    pp = torch.sum(p * p, dim=-1, keepdim=False)
    zz = torch.sum(z * z, dim=-1, keepdim=False)
    pz = torch.sum(p * z, dim=-1, keepdim=False)

    denom = 1 - 2 * c * pz + c ** 2 * zz * pp
    alpha = 1 - 2 * c * pz - c * zz
    beta = 1 + c * pp

    under_root_sum = alpha ** 2 * pp + 2 * alpha * beta * pz + beta ** 2 * zz

    mobius_norm = torch.sqrt(under_root_sum) / denom
    return mobius_norm

def faster_dist(p, z):
    mobius_norm = my_norm(p, z)
    curvature = ball.k

    # another_norm = pasted_norm(p.unsqueeze(1).unsqueeze(1), z.squeeze(1), curvature)
    # print(another_norm.shape)
    # another_norm = another_norm.squeeze(1).squeeze(1).T
    # another_norm = torch.sqrt(another_norm)
    # print(another_norm.shape)
    # print("pasted norm")
    # print(another_norm)

    return 2.0 * artan_k(
        mobius_norm, curvature
    )


dist_funcs = [usual_dist, internal_dist, faster_dist]
# dist_funcs = [geoopt_norm, my_norm]
shape = None
values = None

for dist_func in dist_funcs:
    torch.manual_seed(0)
    num_classes = 200
    batch_size = 2**14
    grad = True
    reps = torch.randn(batch_size, num_classes - 1, device="cpu", requires_grad=grad)
    prototypes = torch.randn(num_classes, 1, num_classes - 1, device="cpu", requires_grad=grad)
    # reps = torch.randn(num_classes - 1, device="cpu", requires_grad=grad)
    # prototypes = torch.randn(num_classes - 1, device="cpu", requires_grad=grad)

    reps = ball.expmap0(reps)
    prototypes = ball.expmap0(prototypes)

    start = time.time()
    distances = dist_func(reps, prototypes)
    finish = time.time()
    if shape is None:
        shape = distances.shape
        values = distances.clone().detach()
        print(distances.shape)
    else:
        assert shape == distances.shape
        if not torch.allclose(values, distances.detach(), rtol=2e-3):
            print(f"{dist_func.__name__} failed")
            print(values)
            print(distances)
            print("diff")
            print(torch.abs(values - distances))
            print(f"max diff: {torch.max(torch.abs(values - distances))}")
            print(f"max relative diff: {torch.max(torch.abs(values - distances) / torch.abs(values))}")
            break
    print(f"time for {dist_func.__name__}: {finish - start}")
