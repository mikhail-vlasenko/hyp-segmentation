import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

part = [
    {
        "name": "SPIN Baseline",
        "score": 0.6076,
        "hyperbolic": False,
    },
    {
        "name": "Full Baseline (Our)",
        "score": 0.69,
        "hyperbolic": False,
    },
    {
        "name": "Eucl. Dim reduced",
        "score": 0.6774,
        "hyperbolic": False,
    },
    {
        "name": "Eucl. Prototypes",
        "score": 0.67225,
        "hyperbolic": False,
    },
    {
        "name": "Hyp. Prototypes",
        "score": 0.68,
        "hyperbolic": True,
    },
]

part = pd.DataFrame(part)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6), dpi=200)
sns.barplot(part, x="name", y="score", hue="hyperbolic")
plt.title("Part-level mIoU")
plt.savefig("part-level.png")

subpart = [
    {
        "name": "SPIN Baseline",
        "score": 0.2425,
        "hyperbolic": False,
    },
    {
        "name": "Full Baseline (Our)",
        "score": 0.2698,
        "hyperbolic": False,
    },
    {
        "name": "Maximally separated",
        "score": 0.27,
        "hyperbolic": True,
    },
    {
        "name": "Eucl. Dim reduced",
        "score": 0.26553,
        "hyperbolic": False,
    },
    {
        "name": "Eucl. Prototypes",  # hierarchical prototypes
        "score": 0.13453,
        "hyperbolic": False,
    },
    {
        "name": "Hyp. Prototypes",  # hierarchical prototypes
        "score": 0.15497,
        "hyperbolic": True,
    }
]

subpart = pd.DataFrame(subpart)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6), dpi=200)
sns.barplot(subpart, x="name", y="score", hue="hyperbolic")
plt.xticks(rotation=15, ha='center')
plt.title("Subpart-level mIoU")
plt.savefig("subpart-level.png")

# todo: hyperbolic metric
