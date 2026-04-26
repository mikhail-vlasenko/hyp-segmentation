import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Create the data - combining original models with new ones
data = {
    'Model': ['LISA (13B)', 'GLaMM', 'GLaMM-FT', 'Ours',
              'HIPIE (R-50)', 'HIPIE (ViT-H)', 'PixelLLM (13B)', 'HALLUMI'],
    'mIoU': [0.1155, 0.11, 0.2456, 0.2676,
             0.0010, 0.0092, 0.1013, 0.1845],
    'Parameters': [13e9, 7e9, 21e9, 73e6,
                   200e6, 800e6, 13e9, 7e9],  # Convert to actual numbers
    'Model_Type': ['General', 'General', 'Fine-tuned', 'Fine-tuned',
                   'General', 'General', 'General', 'Fine-tuned']
}

df = pd.DataFrame(data)
OUTPUT_PATH = Path(__file__).resolve().parent / 'plots' / 'subpart_to_params_with_labels.png'

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})


def add_non_overlapping_labels(ax, data, x_col, y_col, label_col):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    point_coords = ax.transData.transform(data[[x_col, y_col]].to_numpy())
    placed_boxes = []
    log_x = np.log10(data[x_col].to_numpy())
    x_mid = np.median(log_x)
    y_mid = np.median(data[y_col].to_numpy())

    def base_offset(row):
        dx = 5 if np.log10(row[x_col]) <= x_mid else -5
        dy = 3 if row[y_col] <= y_mid else -3
        return dx, dy

    def candidate_offsets(base_dx, base_dy, delta_dx=0, delta_dy=0):
        dx = base_dx + delta_dx
        dy = base_dy + delta_dy
        return [
            (dx, dy),
            (dx, dy - np.sign(dy or 1) * 3),
            (dx + np.sign(dx or 1) * 2, dy),
            (dx - np.sign(dx or 1) * 2, dy),
            (dx, -dy),
            (-dx, dy),
            (-dx, -dy),
            (0, dy + np.sign(dy or 1) * 3),
            (dx + np.sign(dx or 1) * 3, 0),
        ]

    preferred_offsets = {
        'Ours': [(0, 3)],
        'GLaMM-FT': [(1, 3)],
        'HALLUMI': [(1, 3)],
        'LISA (13B)': [(5, 8)],
        'GLaMM': [(-10, -3)],
        'PixelLLM (13B)': [(5, -10)],
        'HIPIE (R-50)': [(-5, 2)],
        'HIPIE (ViT-H)': [(1, -3)],
    }

    for idx, row in data.iterrows():
        label = row[label_col]
        annotation = None
        base_dx, base_dy = base_offset(row)
        preferred_deltas = preferred_offsets.get(label, [(0, 0)])
        offsets = []

        for delta_dx, delta_dy in preferred_deltas:
            offsets.extend(candidate_offsets(base_dx, base_dy, delta_dx, delta_dy))

        for dx, dy in offsets:
            candidate = ax.annotate(
                label,
                (row[x_col], row[y_col]),
                xytext=(dx, dy),
                textcoords='offset points',
                ha='center' if dx == 0 else ('left' if dx > 0 else 'right'),
                va='center' if dy == 0 else ('bottom' if dy > 0 else 'top'),
                fontsize=8,
                fontweight='bold' if label == 'Ours' else 'normal',
            )
            fig.canvas.draw()
            bbox = candidate.get_window_extent(renderer=renderer).expanded(1.03, 1.10)

            overlaps_label = any(bbox.overlaps(other) for other in placed_boxes)
            overlaps_point = False
            for other_idx, (other_x, other_y) in enumerate(point_coords):
                if other_idx == idx:
                    continue
                if bbox.contains(other_x, other_y):
                    overlaps_point = True
                    break

            if overlaps_label or overlaps_point:
                candidate.remove()
                continue

            annotation = candidate
            placed_boxes.append(bbox)
            break

        if annotation is None:
            fallback_dx, fallback_dy = base_offset(row)
            fallback = ax.annotate(
                label,
                (row[x_col], row[y_col]),
                xytext=(fallback_dx, fallback_dy),
                textcoords='offset points',
                ha='center' if fallback_dx == 0 else ('left' if fallback_dx > 0 else 'right'),
                va='center' if fallback_dy == 0 else ('bottom' if fallback_dy > 0 else 'top'),
                fontsize=8,
                fontweight='bold' if label == 'Ours' else 'normal',
            )
            fig.canvas.draw()
            placed_boxes.append(fallback.get_window_extent(renderer=renderer).expanded(1.03, 1.10))


# Create the plot
fig, ax = plt.subplots(figsize=(4.0, 3.5))

# Get colors from RdBu_r palette - red is at the beginning
palette = sns.color_palette("RdBu_r", 10)
custom_colors = {'General': palette[0], 'Fine-tuned': palette[-1]}  # Red for General, Blue for Fine-tuned

# Use seaborn scatterplot with custom palette
sns.scatterplot(
    data=df,
    x='Parameters',
    y='mIoU',
    hue='Model_Type',
    s=28,
    palette=custom_colors,
    ax=ax,
)

# Set x-axis to log scale
ax.set_xscale('log')

# Add labels and title
ax.set_xlabel('# Parameters')
ax.set_ylabel('Subpart mIoU')
# plt.title('Model Performance: Parameters vs mIoU')

# Add grid for better readability
ax.grid(True, alpha=0.25, linewidth=0.6)

# Add legend
ax.legend(loc='lower right', frameon=True, borderpad=0.3, handletextpad=0.4)

# Format x-axis labels to show B/M notation
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.0f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))
ax.margins(x=0.15, y=0.15)

add_non_overlapping_labels(ax, df, 'Parameters', 'mIoU', 'Model')

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
