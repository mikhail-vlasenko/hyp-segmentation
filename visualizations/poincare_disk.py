import numpy as np
import geoopt as gt
import torch
from matplotlib import pyplot as plt

ball = gt.PoincareBall(c=1)
fig, ax = plt.subplots(figsize=(6, 6))

# Create radial gradient background
resolution = 500
x = np.linspace(-1, 1, resolution)
y = np.linspace(-1, 1, resolution)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Mask outside the unit circle
Z = np.ma.masked_where(R > 1, R)

# Plot the gradient background
im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='RdBu_r', alpha=0.75)

# Plot the boundary circle
theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(1 * np.cos(theta), 1 * np.sin(theta),
        color="grey", linewidth=1, alpha=1)


def plot_geodesic(x, y):
    x = ball.expmap0(torch.tensor(x))
    y = ball.expmap0(torch.tensor(y))
    max_norm = 0.985
    if x.norm() > max_norm or y.norm() > max_norm:
        return
    _t = torch.linspace(0, 1, 10)[:, None]
    gv = ball.geodesic(_t, x, y)
    ax.plot(*gv.t().numpy(), color="black", linewidth=0.5, alpha=0.7)

def scatter_hyp(x):
    x = ball.expmap0(torch.tensor(x))
    ax.scatter(x[0].numpy(), x[1].numpy(), color="blue", s=5)


def triangular_tessellation(rows, cols, side_length=1.):
    assert rows == cols
    assert rows % 2 == 1, "Number of rows must be odd for symmetry"
    res = np.zeros((rows, cols, 2))
    height = 3 ** 0.5 / 2  # height of equilateral triangle

    for row in range(rows):
        y = side_length * (row - rows//2) * height
        x_offset = 0.5 if row % 2 else 0.

        for col in range(cols):
            x = side_length * ((col - cols//2) + x_offset)
            res[row, col] = [x, y]

    return res

per_row = 97
mat = triangular_tessellation(per_row, per_row, side_length=0.2)
for row in range(per_row):
    for col in range(per_row):
        if col < per_row - 1:
            plot_geodesic(mat[row, col], mat[row, col + 1])
        if row < per_row - 1:
            plot_geodesic(mat[row, col], mat[row + 1, col])
            if row % 2 == 0 and col > 0:
                plot_geodesic(mat[row, col], mat[row + 1, col - 1])
            elif row % 2 == 1 and col < per_row - 1:
                plot_geodesic(mat[row, col], mat[row + 1, col + 1])

ax.axis('equal')
ax.axis('off')
plt.savefig('poincare_disk.png', dpi=300, bbox_inches='tight')