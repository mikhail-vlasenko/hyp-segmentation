import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Wedge
import geoopt as gt
import torch

# Create Poincaré disk background
ball = gt.PoincareBall(c=1)
fig, ax = plt.subplots(figsize=(8, 8))

# Create radial gradient background
resolution = 500
x = np.linspace(-1, 1, resolution)
y = np.linspace(-1, 1, resolution)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)

# Mask outside the unit circle
Z = np.ma.masked_where(R > 1, R)

# Plot the gradient background
im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='RdBu_r', alpha=0.75)

# Plot the boundary circle
theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(1 * np.cos(theta), 1 * np.sin(theta),
        color="grey", linewidth=1, alpha=1)


def plot_geodesic(x, y):
    x_tensor = ball.expmap0(torch.tensor(x))
    y_tensor = ball.expmap0(torch.tensor(y))
    max_norm = 0.985
    if x_tensor.norm() > max_norm or y_tensor.norm() > max_norm:
        return
    _t = torch.linspace(0, 1, 10)[:, None]
    gv = ball.geodesic(_t, x_tensor, y_tensor)
    ax.plot(*gv.t().numpy(), color="black", linewidth=0.75, alpha=0.7)


def triangular_tessellation(rows, cols, side_length=1.):
    assert rows == cols
    assert rows % 2 == 1, "Number of rows must be odd for symmetry"
    res = np.zeros((rows, cols, 2))
    height = 3 ** 0.5 / 2  # height of equilateral triangle

    for row in range(rows):
        y = side_length * (row - rows // 2) * height
        x_offset = 0.5 if row % 2 else 0.

        for col in range(cols):
            x = side_length * ((col - cols // 2) + x_offset)
            res[row, col] = [x, y]

    return res


# Draw tessellation
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

# Now overlay the sunburst diagram
# Define the three concentric rings with equal width (normalized to unit circle)
inner_radius = 0
middle_inner = 1 / 3
middle_outer = 2 / 3
outer_inner = 2 / 3
outer_outer = 1.0

# Top right quarter: 0 to 90 degrees
theta_start = 0
theta_end = 90

# Inner ring - Biped
wedge = Wedge((0, 0), middle_inner, theta_start, theta_end,
              width=middle_inner - inner_radius,
              facecolor='none', edgecolor='black', linewidth=2.5)
ax.add_patch(wedge)
ax.text(middle_inner / 2 * np.cos(np.pi / 4), middle_inner / 2 * np.sin(np.pi / 4),
        'Biped', ha='center', va='center', fontsize=20, weight='bold')

# Middle ring - Head, Arm, Torso
middle_sections = 3
section_angle = (theta_end - theta_start) / middle_sections

labels_middle = ['Head', 'Arm', 'Torso']

for i, label in enumerate(labels_middle):
    theta_sec_start = theta_start + i * section_angle
    theta_sec_end = theta_start + (i + 1) * section_angle

    wedge = Wedge((0, 0), middle_outer, theta_sec_start, theta_sec_end,
                  width=middle_outer - middle_inner,
                  facecolor='none', edgecolor='black', linewidth=2.5)
    ax.add_patch(wedge)

    mid_angle = np.radians((theta_sec_start + theta_sec_end) / 2)
    mid_radius = (middle_inner + middle_outer) / 2
    ax.text(mid_radius * np.cos(mid_angle), mid_radius * np.sin(mid_angle),
            label, ha='center', va='center', fontsize=20, weight='bold')

# Outer ring
outer_structure = [
    ['Nose', 'Forehead', 'Eyes'],
    ['Shoulders', 'Fingers'],
    ['Chest', 'Abdomen']
]

for i, labels_outer in enumerate(outer_structure):
    theta_sec_start = theta_start + i * section_angle
    theta_sec_end = theta_start + (i + 1) * section_angle

    n_subsections = len(labels_outer)
    subsection_angle = (theta_sec_end - theta_sec_start) / n_subsections

    for j, label in enumerate(labels_outer):
        theta_sub_start = theta_sec_start + j * subsection_angle
        theta_sub_end = theta_sec_start + (j + 1) * subsection_angle

        wedge = Wedge((0, 0), outer_outer, theta_sub_start, theta_sub_end,
                      width=outer_outer - outer_inner,
                      facecolor='none', edgecolor='black', linewidth=2.5)
        ax.add_patch(wedge)

        mid_angle = np.radians((theta_sub_start + theta_sub_end) / 2)
        mid_radius = (outer_inner + outer_outer) / 2
        ax.text(mid_radius * np.cos(mid_angle), mid_radius * np.sin(mid_angle),
                label, ha='center', va='center', fontsize=20, weight='bold')

# Set limits to show only top right quarter
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('poincare_sunburst_quarter.png', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()