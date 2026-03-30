import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# Create the plot
plt.figure(figsize=(4, 3), dpi=300)

# Get colors from RdBu_r palette - red is at the beginning
palette = sns.color_palette("RdBu_r", 10)
custom_colors = {'General': palette[0], 'Fine-tuned': palette[-1]}  # Red for General, Blue for Fine-tuned

# Use seaborn scatterplot with custom palette
sns.scatterplot(data=df, x='Parameters', y='mIoU',
                hue='Model_Type', s=100, palette=custom_colors)

# Set x-axis to log scale
plt.xscale('log')

# Add labels and title
plt.xlabel('# Parameters')
plt.ylabel('Subpart mIoU')
# plt.title('Model Performance: Parameters vs mIoU')

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Add legend
plt.legend(loc='best')

# # Annotate points with model names
# for i, row in df.iterrows():
#     plt.annotate(row['Model'],
#                 (row['Parameters'], row['mIoU']),
#                 xytext=(5, 5), textcoords='offset points',
#                 fontsize=9, alpha=0.8)

# Format x-axis labels to show B/M notation
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.0f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))

plt.tight_layout()
plt.savefig('plots/subpart_to_params.png', dpi=300, bbox_inches='tight')
