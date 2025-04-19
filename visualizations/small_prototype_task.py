import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
from geoopt import PoincareBall
from lightning import seed_everything
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd

from hyperbolic_layers import fast_dist


class Synthetic2DClassificationDataset(Dataset):
    def __init__(self, n_samples_per_class=1000):
        # Means for 3 classes: classes 0 and 1 are close, class 2 is separated.
        self.means = torch.tensor([
            [0.0, 0.0],  # Class 0
            [0.5, 0.5],  # Class 1 (very similar to Class 0)
            [5.0, 5.0]  # Class 2 (well separated)
        ], dtype=torch.float32)

        # Covariance matrices for each Gaussian
        self.covariances = torch.tensor([
            [[0.1, 0.0], [0.0, 0.1]],
            [[0.1, 0.0], [0.0, 0.1]],
            [[0.1, 0.0], [0.0, 0.1]]
        ], dtype=torch.float32)

        data = []
        labels = []
        for label, (mean, cov) in enumerate(zip(self.means, self.covariances)):
            dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
            samples = dist.sample((n_samples_per_class,))
            data.append(samples)
            labels.append(torch.full((n_samples_per_class,), label, dtype=torch.long))
        self.data = torch.cat(data, dim=0)
        self.labels = torch.cat(labels, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SmallNet(nn.Module):
    def __init__(self, hyperbolic, c=1.0, tau=1.0, num_classes=3, hidden_dim=4, out_dim=2, prototypes=None):
        super().__init__()
        self.hyperbolic = hyperbolic
        self.c = torch.tensor(c)
        self.tau = tau
        self.num_classes = num_classes
        self.out_dim = out_dim

        # A simple MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
        )
        self.prototypes = prototypes
        if self.prototypes is not None:
            num_classes = self.prototypes.shape[1]

        self.classifier = nn.Linear(self.out_dim, num_classes)
        # hyperbolic without prototypes is not supported
        assert not (self.hyperbolic and self.prototypes is None), "Hyperbolic model requires prototypes."

        if self.hyperbolic:
            self.prototypes = self.prototypes.unsqueeze(1) * 0.95

        self.ball = PoincareBall(c=self.c)

    def forward(self, x, return_repr=False):
        """
        Args:
            x: input tensor of shape (batch_size, 2)
            return_repr: if True, also return the learned representation.
        Returns:
            logits: unnormalized log-probabilities for each class.
        """
        hidden = self.encoder(x)  # shape: (batch, hidden_dim)
        if self.hyperbolic:
            hidden = self.classifier(hidden)
            # Map Euclidean features to the hyperbolic space.
            rep = self.ball.expmap0(hidden)
            # Compute hyperbolic distances to prototypes.
            distances = fast_dist(rep, self.prototypes, self.c).T
            assert not torch.isnan(distances).any()
            # Lower distance should mean a higher logit; hence, we use negative distances.
            logits = - self.tau * distances
        else:
            logits = self.classifier(hidden)
            if self.prototypes is not None:
                logits = torch.einsum("bc,nc->bn", logits, self.prototypes)
            rep = hidden
        if return_repr:
            return logits, rep
        return logits


def train_model(model, dataloader, num_epochs=50, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    epoch_acc = 0.0
    losses_per_epoch = []
    new_losses_per_epoch = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        new_running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            if model.prototypes is not None:
                # if model.hyperbolic:
                #     # Use the hyperbolic distance to the prototypes as logits.
                distances = -logits
                additional_loss_weight = 1
                # add a loss term that is (dist to correct)/(dist to closest incorrect)
                batch_size = inputs.size(0)
                correct_dists = distances[torch.arange(batch_size), labels]
                # Create a mask to select distances corresponding to incorrect prototypes.
                mask = torch.ones_like(distances, dtype=torch.bool)
                mask[torch.arange(batch_size), labels] = False
                # Find, for each sample, the minimum distance among incorrect prototypes.
                min_incorrect = distances.masked_select(mask).view(batch_size, -1).min(dim=1)[0]
                eps = 1e-4  # small value to prevent division by zero
                min_incorrect = torch.max(min_incorrect, torch.full_like(min_incorrect, eps))
                correct_dists = torch.max((correct_dists * 2) - min_incorrect, torch.full_like(min_incorrect, 0))
                # Compute the ratio loss: lower when the correct distance is much smaller than the best incorrect distance.
                ratio_loss = (correct_dists / min_incorrect)
                ratio_loss = torch.mean(ratio_loss)
                assert not torch.isnan(ratio_loss)
                new_running_loss += ratio_loss.item() * inputs.size(0)
                # Add the additional loss to the cross-entropy loss.
                loss += additional_loss_weight * ratio_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        losses_per_epoch.append(epoch_loss)
        new_losses_per_epoch.append(new_running_loss / total)
        # print(f"Epoch {epoch + 1}/{num_epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
    return epoch_acc, losses_per_epoch, new_losses_per_epoch

def make_prototypes(close_prototypes):
    if close_prototypes:
        # A and B are close, C is far away.
        angles = [0, np.pi / 20, np.pi]
    else:
        # Equally spaced prototypes in the Poincaré ball.
        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]

    prototypes = torch.tensor([
        [np.cos(angle), np.sin(angle)] for angle in angles
    ], dtype=torch.float32)  # Shape: (num_classes, out_dim)
    return prototypes

def evaluate_configuration(hyperbolic, use_prototypes, close_prototypes=True):
    dataset = Synthetic2DClassificationDataset(n_samples_per_class=1000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    prototypes = make_prototypes(close_prototypes)
    losses1, losses2 = [], []

    accs = []
    for i in range(NUM_RUNS):
        model = SmallNet(
            hyperbolic=hyperbolic,
            c=1.0,
            tau=15.0,
            prototypes=prototypes if use_prototypes else None,
        )

        accuracy, losses_per_epoch, new_losses_per_epoch = train_model(model, dataloader)
        accs.append(accuracy)
        losses1.append(losses_per_epoch)
        losses2.append(new_losses_per_epoch)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(np.array(losses1).mean(0), label="Cross-Entropy Loss")
    plt.plot(np.array(losses2).mean(0), label="Ratio Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Losses per Epoch. {hyperbolic=}, {use_prototypes=}, {close_prototypes=}")
    plt.legend()
    plt.savefig(f"plots/losses_{hyperbolic}_{use_prototypes}_{close_prototypes}.png")
    prefix = f"Hyperbolic" if hyperbolic else "Euclidean"
    prefix += f" Prototypical (A and B are {'close' if close_prototypes else 'far'})" if use_prototypes else ""
    title = f"{prefix}. Acc = {np.mean(accs):.3f}+-{np.std(accs, ddof=1):.4f}"
    print(title)
    return np.mean(accs), np.std(accs, ddof=1)

def plot_decision_boundary(model, dataset, hyperbolic, accuracy, close_prototypes, device="cpu"):
    model.eval()
    # Create a grid of points covering the data domain.
    x_min, x_max = dataset.data[:, 0].min() - 1, dataset.data[:, 0].max() + 1
    y_min, y_max = dataset.data[:, 1].min() - 1, dataset.data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
    with torch.no_grad():
        logits = model(grid)
        preds = torch.argmax(logits, dim=1).reshape(xx.shape)
    plt.figure(figsize=(8, 8), dpi=100)
    plt.contourf(xx, yy, preds.cpu().numpy(), alpha=0.3, cmap="coolwarm")
    plt.scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.labels, edgecolors="k", cmap="coolwarm")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def plot_dataset_with_prototype_versions(dataset, device="cpu"):
    sns.set(style="whitegrid")

    # Generate prototypes for both cases using the provided function.
    prototypes_close = make_prototypes(close_prototypes=True)
    prototypes_far = make_prototypes(close_prototypes=False)

    # Convert dataset tensors to NumPy arrays.
    data = dataset.data.cpu().detach().numpy()
    labels = dataset.labels.cpu().detach().numpy()

    # Prepare a figure with 1 row and 3 columns.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=200)

    num_classes = int(np.max(labels)) + 1
    cmap = cm.get_cmap("Accent", num_classes)
    colors = [cmap(i) for i in range(num_classes)]

    # Subplot 1: Plot the dataset.
    scatter = axs[0].scatter(data[:, 0], data[:, 1], c=labels, cmap="Accent", edgecolors="k", alpha=0.6)
    axs[0].set_title("Dataset")
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")

    # Function to draw a disk (unit circle) and the prototypes as vectors from the origin,
    # colored according to class.
    def plot_prototypes(ax, prototypes, title):
        # Draw a circle representing the unit disk (Poincaré ball).
        theta = np.linspace(0, 2 * np.pi, 400)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(circle_x, circle_y, color="gray", linestyle="--")
        if prototypes.dim() == 3:
            prototypes_mod = prototypes.squeeze(1)
        else:
            prototypes_mod = prototypes
        prototypes_np = prototypes_mod.cpu().detach().numpy()
        # Draw arrows using annotate with increased line width for brightness.
        for idx, proto in enumerate(prototypes_np):
            ax.annotate(
                '',
                xy=(proto[0], proto[1]),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle='->',
                    color=colors[idx],
                    lw=3
                )
            )
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal', 'box')
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Subplot 2: Plot the 'Close Prototypes' on a disk.
    plot_prototypes(axs[1], prototypes_close, "Close Prototypes")

    # Subplot 3: Plot the 'Far Prototypes' on a disk.
    plot_prototypes(axs[2], prototypes_far, "Far Prototypes")

    plt.tight_layout()
    plt.savefig("synthetic_dataset_with_prototypes.png")


if __name__ == "__main__":
    NUM_RUNS = 10
    seed_everything(42)

    dataset = Synthetic2DClassificationDataset(n_samples_per_class=1000)
    # plot_dataset_with_prototype_versions(dataset)

    results = []

    mean, std = evaluate_configuration(hyperbolic=True, use_prototypes=True, close_prototypes=True)
    results.append({
        "Method": "Hyp, CLOSE",
        "Mean": mean,
        "Std": std
    })

    mean, std = evaluate_configuration(hyperbolic=True, use_prototypes=True, close_prototypes=False)
    results.append({
        "Method": "Hyp, FAR",
        "Mean": mean,
        "Std": std
    })

    mean, std = evaluate_configuration(hyperbolic=False, use_prototypes=True, close_prototypes=True)
    results.append({
        "Method": "Euc, CLOSE",
        "Mean": mean,
        "Std": std
    })

    mean, std = evaluate_configuration(hyperbolic=False, use_prototypes=True, close_prototypes=False)
    results.append({
        "Method": "Euc, FAR",
        "Mean": mean,
        "Std": std
    })

    mean, std = evaluate_configuration(hyperbolic=False, use_prototypes=False)
    results.append({
        "Method": "Euc, BASELINE",
        "Mean": mean,
        "Std": std
    })

    # Create a DataFrame with the results.
    df = pd.DataFrame(results)

    # Create the bar plot.
    plt.figure(figsize=(10, 6), dpi=200)
    ax = sns.barplot(x="Method", y="Mean", data=df)

    # Add error bars using matplotlib.
    for idx, row in df.iterrows():
        ax.errorbar(idx, row["Mean"], yerr=row["Std"], fmt='none', c='black', capsize=5)

    # Customize labels and rotation for clarity.
    plt.title("Evaluation of Configurations")
    plt.ylabel("Accuracy")
    plt.xlabel("Method")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("synthetic_results_bar_chart.png")

    # Visualize the decision boundaries.
    # plot_decision_boundary(model, dataset, hyperbolic=True, accuracy=accuracy, close_prototypes=close_prototypes)
