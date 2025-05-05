import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
from geoopt import PoincareBall
from lightning import seed_everything
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd

from hyperbolic_layers import fast_dist
from model import PrototypeRatioLoss


class Synthetic2DClassificationDataset(Dataset):
    def __init__(self, n_samples_per_class=1000):
        # Means for 3 classes: classes 0 and 1 are close, class 2 is separated.
        self.means = torch.tensor([
            [0.0, 0.0],  # Class 0
            [0.5, 0.5],  # Class 1 (very similar to Class 0)
            [5.0, 5.0]   # Class 2 (well separated)
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
        hidden = self.encoder(x)
        if self.hyperbolic:
            hidden = self.classifier(hidden)
            rep = self.ball.expmap0(hidden)
            distances = fast_dist(rep, self.prototypes, self.c).T
            # print(torch.norm(hidden, p=2, dim=1).max().item(), torch.norm(rep, p=2, dim=1).max().item())
            if torch.isnan(distances).any():
                print("nans")
                print(torch.norm(hidden, p=2, dim=1).max().item(), torch.norm(rep, p=2, dim=1).max().item())
                print(torch.isnan(rep).any())
                print(torch.isnan(hidden).any())
                raise ValueError("NaN in distances")
            logits = - self.tau * distances
        else:
            logits = self.classifier(hidden)
            if self.prototypes is not None:
                logits = torch.einsum("bc,nc->bn", logits, self.prototypes)
            rep = hidden

        if return_repr:
            return logits, rep
        return logits


def hinge_norm_penalty(h, threshold=0.5):
    sqnorm = h.pow(2).sum(dim=1)
    over = F.relu(sqnorm - threshold)
    return (over**2).mean()

def train_model(model, dataloader, num_epochs=50, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
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

            logits, rep = model(inputs, return_repr=True)
            loss = criterion(logits, labels)
            if model.prototypes is not None:
                loss_fct = PrototypeRatioLoss(weight=1.0)
                new_loss = loss_fct(logits, labels)
                new_running_loss += new_loss.item() * inputs.size(0)
                loss = loss + new_loss
                norm_penalty = hinge_norm_penalty(rep)
                loss = loss + norm_penalty

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

        losses_per_epoch.append(running_loss / total)
        new_losses_per_epoch.append(new_running_loss / total)

    print("Training complete.")
    accuracy = correct / total
    return accuracy, losses_per_epoch, new_losses_per_epoch


def make_prototypes(close_prototypes):
    if close_prototypes:
        angles = [0, np.pi / 20, np.pi]
    else:
        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]

    prototypes = torch.tensor([
        [np.cos(angle), np.sin(angle)] for angle in angles
    ], dtype=torch.float32)
    return prototypes


def evaluate_configuration(hyperbolic, use_prototypes, close_prototypes=True):
    # create full dataset, then split 80% train / 20% test
    full_ds = Synthetic2DClassificationDataset(n_samples_per_class=1000)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_ds, test_ds = random_split(full_ds, [n_train, n_test])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    prototypes = make_prototypes(close_prototypes)
    losses1, losses2 = [], []
    train_accs, test_accs = [], []

    for _ in range(NUM_RUNS):
        model = SmallNet(
            hyperbolic=hyperbolic,
            c=1.0,
            tau=15.0,
            prototypes=prototypes if use_prototypes else None,
        )

        # Train
        tr_acc, losses_per_epoch, new_losses_per_epoch = train_model(model, train_loader)
        train_accs.append(tr_acc)
        losses1.append(losses_per_epoch)
        losses2.append(new_losses_per_epoch)

        # Test
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_t, y_t in test_loader:
                logits = model(x_t)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_t).sum().item()
                total += y_t.size(0)
        test_accs.append(correct / total)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(np.array(losses1).mean(0), label="Cross-Entropy Loss")
    plt.plot(np.array(losses2).mean(0), label="Ratio Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Losses per Epoch. hyperbolic={hyperbolic}, use_prototypes={use_prototypes}, close_prototypes={close_prototypes}")
    plt.legend()
    plt.savefig(f"plots/losses_{hyperbolic}_{use_prototypes}_{close_prototypes}.png")

    prefix = f"{'Hyperbolic' if hyperbolic else 'Euclidean'}"
    if use_prototypes:
        prefix += f" Prototypical (A and B are {'close' if close_prototypes else 'far'})"

    tr_mean, tr_std = np.mean(train_accs), np.std(train_accs, ddof=1)
    te_mean, te_std = np.mean(test_accs),  np.std(test_accs,  ddof=1)
    title = (f"{prefix}. "
             f"Train Acc = {tr_mean:.3f}±{tr_std:.4f}, "
             f"Test Acc = {te_mean:.3f}±{te_std:.4f}")
    print(title)

    return tr_mean, tr_std, te_mean, te_std


if __name__ == "__main__":
    NUM_RUNS = 10
    seed_everything(42)

    results = []

    for hyp, use_proto, close in [
        (True,  True,  True),
        (True,  True,  False),
        (False, True,  True),
        (False, True,  False),
        (False, False, False),
    ]:
        tr_mean, tr_std, te_mean, te_std = evaluate_configuration(
            hyperbolic=hyp,
            use_prototypes=use_proto,
            close_prototypes=close
        )
        method = (
            f"{'Hyp' if hyp else 'Euc'}, "
            f"{'CLOSE' if close else 'FAR' if use_proto else 'BASELINE'}"
        )
        results.append({
            "Method":     method,
            "Train Mean": tr_mean,
            "Train Std":  tr_std,
            "Test Mean":  te_mean,
            "Test Std":   te_std,
        })

    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6), dpi=200)
    df_melt = df.melt(id_vars="Method", value_vars=["Train Mean","Test Mean"],
                      var_name="Split", value_name="Accuracy")
    ax = sns.barplot(x="Method", y="Accuracy", hue="Split", data=df_melt)

    for idx, row in df.iterrows():
        ax.errorbar(idx - 0.2, row["Train Mean"], yerr=row["Train Std"],
                    fmt='none', c='black', capsize=5)
        ax.errorbar(idx + 0.2, row["Test Mean"],  yerr=row["Test Std"],
                    fmt='none', c='black', capsize=5)

    plt.title("Evaluation of Configurations (Train vs Test)")
    plt.ylabel("Accuracy")
    plt.xlabel("Method")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("synthetic_results_bar_chart.png")
