import time

import geoopt as gt
import torch
from geoopt.manifolds.stereographic.math import artan_k, _mobius_add
import torch.nn.functional as F


ball = gt.PoincareBall(c=0.5)


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

    B = 1 - c * pp  # sh ch ## beta = B/D

    D = 1 + torch.add(2 * c * px, sqsq)  # sh B,H,W,ch
    D = torch.maximum(D, torch.tensor(EPS))

    # calculate mobadd norm indepently from mob add
    # if mob_add = alpha * p + beta * x, then
    #  |mob_add|^2 = theta**2 * |p|^2 + gamma^2 * |x|^2 + 2*theta*gamma*|px|
    # theta = A/D, gamma = B/D
    alpha = A / D  # B,H,W,ch
    beta = B[None, None, None, :] / D  # B,H,W,ch

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
    return _mobius_add(p, z, c, dim=dim).norm(dim=dim, p=2, keepdim=keepdim)

def atigh_2022_norm(p, z):
    # based on Atigh et al. 2022
    # the formulas expect negative curvature
    c = -ball.k
    pp = torch.sum(p * p, dim=-1, keepdim=False)
    zz = torch.sum(z * z, dim=-1, keepdim=False)
    pz = torch.sum(p * z, dim=-1, keepdim=False)

    # compute the denominator for alpha and beta from Eq. 9
    denom = 1 + 2 * c * pz + c ** 2 * zz * pp
    denom = denom.clamp_min(1e-15)

    # compute the numerators for alpha and beta from Eq. 9
    a_num = 1 + 2 * c * pz + c * zz
    b_num = 1 - c * pp

    alpha = a_num / denom
    beta = b_num / denom

    # by Eq. 11
    mobius_norm2 = alpha ** 2 * pp + 2 * alpha * beta * pz + beta ** 2 * zz
    mobius_norm = torch.sqrt(mobius_norm2)
    return mobius_norm

def atigh_2022_dist(p, z):
    mobius_norm = atigh_2022_norm(-p, z)
    curvature = ball.k

    return 2.0 * artan_k(
        mobius_norm, curvature
    )

def with_pasted_norm(p, z):
    c = -ball.k  # take the positive curvature here
    mobaddnorm = pasted_norm(p.unsqueeze(1).unsqueeze(1), z.squeeze(1), c)
    mobaddnorm = mobaddnorm.squeeze(1).squeeze(1).T
    return 2.0 * artan_k(
        torch.sqrt(mobaddnorm), ball.k
    )

def arcosh(x):
    # there isn't an arcosh in geoopt
    return (x + torch.sqrt(-1 + x.pow(2))).clamp_min(1e-15).log().to(x.dtype)

def induced_distance(p, z):
    # https://math.stackexchange.com/questions/3279762/emulating-distance-on-poincar%C3%A9-disk-for-different-curvatures
    r = 1 / torch.sqrt(-ball.k)
    p = p / r
    z = z / r

    # we need to compute
    # ∥p−z∥^2
    # a way to do that without a memory blowup is
    # ∥p∥^2 + ∥z∥^2 − 2⟨z,p⟩
    pp = torch.sum(p.pow(2), dim=-1, keepdim=False)
    zz = torch.sum(z.pow(2), dim=-1, keepdim=False)
    pz = torch.einsum('...i,...i->...', p, z)

    num = pp + zz - 2 * pz

    # conveniently, we can reuse the above computation to get the denominator
    denom = ((1 - pp) * (1 - zz))
    return r * arcosh(
        1 + 2 * num / denom,
    )

def my_norm(p, z):
    c = ball.k  # this actually gives negative curvature
    pp = torch.sum(p.pow(2), dim=-1, keepdim=False)
    zz = torch.sum(z.pow(2), dim=-1, keepdim=False)
    pz = torch.einsum('...i,...i->...', p, z)

    denom = 1 - 2 * c * pz + c ** 2 * zz * pp
    alpha = 1 - 2 * c * pz - c * zz
    beta = 1 + c * pp

    under_root_sum = alpha ** 2 * pp + 2 * alpha * beta * pz + beta ** 2 * zz

    mobius_norm = torch.sqrt(under_root_sum) / denom.clamp_min(1e-15)
    return mobius_norm

def faster_dist(p, z):
    mobius_norm = my_norm(-p, z)
    curvature = ball.k

    return 2.0 * artan_k(
        mobius_norm, curvature
    )


dist_funcs = [usual_dist, induced_distance, faster_dist, atigh_2022_dist, with_pasted_norm]
# dist_funcs = [geoopt_norm, my_norm, atigh_2022_norm]
shape = None
values = None

for dist_func in dist_funcs:
    torch.manual_seed(0)
    num_classes = 204
    batch_size = 2**14
    grad = True
    reps = torch.randn(batch_size, num_classes - 1, device="cuda", requires_grad=grad)
    prototypes = torch.randn(num_classes, 1, num_classes - 1, device="cuda", requires_grad=grad)
    # reps = torch.randn(num_classes - 1, device="cpu", requires_grad=grad)
    # prototypes = torch.randn(num_classes - 1, device="cpu", requires_grad=grad)

    reps = ball.expmap0(reps)
    prototypes = ball.expmap0(prototypes)

    start = time.time()
    distances = dist_func(reps, prototypes).T
    finish = time.time()
    if shape is None:
        shape = distances.shape
        values = distances.clone().detach()
        print(distances.shape)
    else:
        assert shape == distances.shape
        if not torch.allclose(values, distances.detach(), rtol=3e-3):
            print(f"{dist_func.__name__} failed by getting")
            print(distances)
            print("expected")
            print(values)
            print("diff")
            print(torch.abs(values - distances))
            print(f"max diff: {torch.max(torch.abs(values - distances))}")
            print(f"max relative diff: {torch.max(torch.abs(values - distances) / torch.abs(values))}")
            break
    print(f"time for {dist_func.__name__}: {finish - start}. "
          f"max relative diff: {torch.max(torch.abs(values - distances) / torch.abs(values))}")
