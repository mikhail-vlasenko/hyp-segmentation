import numpy as np
import pandas as pd

# Read the CSV file
df = pd.read_csv("wandb_export_2025-03-26T11_19_00.427+01_00.csv")

# Create a performance column: use test_mIoU_subpart if available, else use val_mIoU_subpart.
df['performance'] = df['test_mIoU_subpart']
df['from_val'] = False
mask = df['performance'].isna()
df.loc[mask, 'performance'] = df.loc[mask, 'val_mIoU_subpart']
df.loc[mask, 'from_val'] = True

# Format the performance metric as a string with an asterisk if it came from the validation set.
df['performance_str'] = df.apply(lambda row: f"{row['performance']:.6f}{' (on val)' if row['from_val'] else ''}", axis=1)

def focal_loss(row):
    if row['focal_loss'] and not isinstance(row['focal_loss'], type(np.nan)):
        return True
    return False

def fix_gamma(row):
    if not row['focal_loss']:
        return "N/A"
    return row['focal_loss_gamma']

def fix_bg_weight(row):
    if row['focal_loss']:
        return "N/A"
    return row['background_loss_weight']

def fix_tau(row):
    if row['hyperbolic'] and row['max_class_sep']:
        return row['tau']
    return "N/A"

df['focal_loss'] = df.apply(focal_loss, axis=1)
df['focal_loss_gamma'] = df.apply(fix_gamma, axis=1)
df['background_loss_weight'] = df.apply(fix_bg_weight, axis=1)
df['tau'] = df.apply(fix_tau, axis=1)

# Rename max_class_sep to prototypes for clarity.
df = df.rename(columns={"max_class_sep": "prototypes"})

# Group by the desired parameters and calculate the average, sample std, and count for performance.
grouped = df.groupby(
    ['hyperbolic', 'prototypes', 'focal_loss', 'focal_loss_gamma', 'background_loss_weight', 'tau']
)['performance'].agg(['mean', 'std', 'count']).reset_index()

# Format the performance mean.
grouped['mean'] = grouped['mean'].apply(lambda x: f"{x:.4f}")

# Format the std with the count appended.
grouped['std'] = grouped.apply(
    lambda row: f"{row['std']:.4f} (from {int(row['count'])} run{'s' if row['count'] != 1 else ''})"
                if pd.notna(row['std']) else f"only 1 run",
    axis=1)
grouped.drop(columns='count', inplace=True)

grouped.sort_values(by='mean', ascending=False, inplace=True)

# Optionally, you can rename the columns for better display.
grouped = grouped.rename(columns={
    'hyperbolic': 'Hyperbolic',
    'prototypes': 'Prototypes',
    'focal_loss': 'Is Focal Loss',
    'focal_loss_gamma': 'Focal Gamma',
    'background_loss_weight': 'BG Weight in CE loss',
    'tau': 'Prototypes Temp',
    'mean': 'Avg Test mIoU',
    'std': 'Std mIoU'
})

# Display the grouped table as markdown.
print(grouped.to_markdown(index=False))
