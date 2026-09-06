import os
import glob
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "PointNetDataset"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

NUM_CLASSES = 7

# RTX 3050 6GB
NUM_POINTS = 2048
BATCH_SIZE = 8

EPOCHS = 25
LEARNING_RATE = 0.001

NUM_WORKERS = 2

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("POINTNET++ SEMANTIC SEGMENTATION TRAINING")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ============================================================
# DATASET
# ============================================================

class LidarDataset(Dataset):

    def __init__(self, directory, num_points=2048):

        self.files = sorted(
            glob.glob(os.path.join(directory, "*.npz"))
        )

        self.num_points = num_points

        print(
            f"Loaded {len(self.files)} files from {directory}"
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No NPZ files found in {directory}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_path = self.files[index]

        data = np.load(file_path)

        points = data["points"].astype(
            np.float32
        )

        labels = data["labels"].astype(
            np.int64
        )

        # ----------------------------------------------------
        # Convert labels 1-7 → 0-6
        # ----------------------------------------------------

        labels = labels - 1

        # ----------------------------------------------------
        # Remove invalid labels
        # ----------------------------------------------------

        valid = (
            (labels >= 0) &
            (labels < NUM_CLASSES)
        )

        points = points[valid]
        labels = labels[valid]

        # ----------------------------------------------------
        # Random sampling
        # ----------------------------------------------------

        n = len(points)

        if n >= self.num_points:

            indices = np.random.choice(
                n,
                self.num_points,
                replace=False
            )

        else:

            indices = np.random.choice(
                n,
                self.num_points,
                replace=True
            )

        points = points[indices]
        labels = labels[indices]

        # ----------------------------------------------------
        # Normalize point cloud (xyz coordinates only)
        # ----------------------------------------------------

        xyz = points[:, :3]

        centroid = np.mean(
            xyz,
            axis=0,
            keepdims=True
        )

        xyz = xyz - centroid

        scale = np.max(
            np.linalg.norm(xyz, axis=1)
        )

        if scale > 0:
            xyz = xyz / scale

        return (
            torch.from_numpy(xyz),
            torch.from_numpy(labels)
        )


# ============================================================
# FARTHEST POINT SAMPLING
# ============================================================

def farthest_point_sample(x, npoint):

    """
    x:
        B x N x 3

    return:
        B x npoint
    """

    B, N, C = x.shape

    device = x.device

    centroids = torch.zeros(
        B,
        npoint,
        dtype=torch.long,
        device=device
    )

    distance = torch.full(
        (B, N),
        1e10,
        dtype=x.dtype,
        device=device
    )

    farthest = torch.randint(
        0,
        N,
        (B,),
        dtype=torch.long,
        device=device
    )

    batch_indices = torch.arange(
        B,
        dtype=torch.long,
        device=device
    )

    for i in range(npoint):

        centroids[:, i] = farthest

        centroid = x[
            batch_indices,
            farthest,
            :
        ].view(B, 1, C)

        dist = torch.sum(
            (x - centroid) ** 2,
            dim=-1
        )

        distance = torch.minimum(
            distance,
            dist
        )

        farthest = torch.argmax(
            distance,
            dim=-1
        )

    return centroids


# ============================================================
# INDEX POINTS
# ============================================================

def index_points(points, idx):

    """
    points:
        B x N x C
    idx:
        B x S or B x S1 x S2 ...
    return:
        indexed points of shape (B, *idx.shape[1:], C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# ============================================================
# KNN GROUPING
# ============================================================

def knn_group(xyz, centers, k):

    """

    xyz:
        B x N x 3

    centers:
        B x S x 3

    returns:
        B x S x K x 3
    """

    distances = torch.cdist(
        centers,
        xyz
    )

    _, indices = torch.topk(
        distances,
        k=k,
        dim=-1,
        largest=False
    )

    return index_points(xyz, indices)


# ============================================================
# POINTNET SET ABSTRACTION
# ============================================================

class SetAbstraction(nn.Module):

    def __init__(
        self,
        npoint,
        k,
        in_channels,
        mlp
    ):

        super().__init__()

        self.npoint = npoint
        self.k = k

        layers = []

        last_channel = in_channels

        for channel in mlp:

            layers.append(
                nn.Conv2d(
                    last_channel,
                    channel,
                    1
                )
            )

            layers.append(
                nn.BatchNorm2d(channel)
            )

            layers.append(
                nn.ReLU()
            )

            last_channel = channel

        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features=None):

        # ----------------------------------------------------
        # Sample centers
        # ----------------------------------------------------

        sample_idx = farthest_point_sample(
            xyz,
            self.npoint
        )

        centers = index_points(
            xyz,
            sample_idx
        )

        # ----------------------------------------------------
        # Group neighbors
        # ----------------------------------------------------

        grouped_xyz = knn_group(
            xyz,
            centers,
            self.k
        )

        # ----------------------------------------------------
        # Local coordinates
        # ----------------------------------------------------

        grouped_xyz = (
            grouped_xyz -
            centers.unsqueeze(2)
        )

        # ----------------------------------------------------
        # Add additional features
        # ----------------------------------------------------

        if features is not None:

            B, N, C = features.shape

            idx = torch.cdist(
                centers,
                xyz
            ).topk(
                self.k,
                dim=-1,
                largest=False
            )[1]

            grouped_features = index_points(
                features,
                idx
            )

            new_features = torch.cat(
                [
                    grouped_xyz,
                    grouped_features
                ],
                dim=-1
            )

        else:

            new_features = grouped_xyz

        # ----------------------------------------------------
        # PointNet local feature extraction
        # ----------------------------------------------------

        new_features = new_features.permute(
            0,
            3,
            1,
            2
        )

        new_features = self.mlp(
            new_features
        )

        # ----------------------------------------------------
        # Max pooling
        # ----------------------------------------------------

        new_features = torch.max(
            new_features,
            dim=-1
        )[0]

        new_features = new_features.permute(
            0,
            2,
            1
        )

        return centers, new_features


# ============================================================
# FEATURE PROPAGATION
# ============================================================

class FeaturePropagation(nn.Module):

    def __init__(
        self,
        in_channels,
        mlp
    ):

        super().__init__()

        layers = []

        last_channel = in_channels

        for channel in mlp:

            layers.append(
                nn.Conv1d(
                    last_channel,
                    channel,
                    1
                )
            )

            layers.append(
                nn.BatchNorm1d(channel)
            )

            layers.append(
                nn.ReLU()
            )

            last_channel = channel

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz1,
        xyz2,
        features1,
        features2
    ):

        # xyz1 = dense points
        # xyz2 = sampled points

        distances = torch.cdist(
            xyz1,
            xyz2
        )

        dists, idx = torch.topk(
            distances,
            3,
            dim=-1,
            largest=False
        )

        dists = torch.clamp(
            dists,
            min=1e-10
        )

        weights = 1.0 / dists

        weights = weights / torch.sum(
            weights,
            dim=-1,
            keepdim=True
        )

        interpolated = index_points(
            features2,
            idx
        )

        interpolated = torch.sum(
            interpolated *
            weights.unsqueeze(-1),
            dim=2
        )

        if features1 is not None:

            new_features = torch.cat(
                [
                    features1,
                    interpolated
                ],
                dim=-1
            )

        else:

            new_features = interpolated

        new_features = new_features.permute(
            0,
            2,
            1
        )

        new_features = self.mlp(
            new_features
        )

        new_features = new_features.permute(
            0,
            2,
            1
        )

        return new_features


# ============================================================
# POINTNET++ SEMANTIC SEGMENTATION MODEL
# ============================================================

class PointNet2Segmentation(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Set Abstraction layers
        # ----------------------------------------------------

        self.sa1 = SetAbstraction(
            npoint=512,
            k=16,
            in_channels=3,
            mlp=[64, 64, 128]
        )

        self.sa2 = SetAbstraction(
            npoint=256,
            k=16,
            in_channels=131,
            mlp=[128, 128, 256]
        )

        self.sa3 = SetAbstraction(
            npoint=64,
            k=16,
            in_channels=259,
            mlp=[256, 512, 1024]
        )

        # ----------------------------------------------------
        # Feature propagation
        # ----------------------------------------------------

        self.fp3 = FeaturePropagation(
            256 + 1024,
            [256, 256]
        )

        self.fp2 = FeaturePropagation(
            128 + 256,
            [256, 128]
        )

        self.fp1 = FeaturePropagation(
            3 + 128,
            [128, 128, 128]
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Conv1d(
                128,
                128,
                1
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Conv1d(
                128,
                NUM_CLASSES,
                1
            )
        )

    def forward(self, xyz):

        # ----------------------------------------------------
        # SA1
        # ----------------------------------------------------

        xyz1, feat1 = self.sa1(
            xyz
        )

        # ----------------------------------------------------
        # SA2
        # ----------------------------------------------------

        xyz2, feat2 = self.sa2(
            xyz1,
            feat1
        )

        # ----------------------------------------------------
        # SA3
        # ----------------------------------------------------

        xyz3, feat3 = self.sa3(
            xyz2,
            feat2
        )

        # ----------------------------------------------------
        # Feature propagation
        # ----------------------------------------------------

        feat2_up = self.fp3(
            xyz2,
            xyz3,
            feat2,
            feat3
        )

        feat1_up = self.fp2(
            xyz1,
            xyz2,
            feat1,
            feat2_up
        )

        feat0_up = self.fp1(
            xyz,
            xyz1,
            xyz,
            feat1_up
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            feat0_up.permute(
                0,
                2,
                1
            )
        )

        return logits.permute(
            0,
            2,
            1
        )

if __name__ == "__main__":

    # ============================================================
    # DATA LOADERS
    # ============================================================

    print("\nLoading datasets...")

    train_dataset = LidarDataset(
        TRAIN_DIR,
        NUM_POINTS
    )

    val_dataset = LidarDataset(
        VAL_DIR,
        NUM_POINTS
    )

    test_dataset = LidarDataset(
        TEST_DIR,
        NUM_POINTS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )


    # ============================================================
    # MODEL
    # ============================================================

    print("\nCreating PointNet++ model...")

    model = PointNet2Segmentation().to(
        DEVICE
    )

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


    # ============================================================
    # CLASS WEIGHTS
    # ============================================================

    # Classes:
    #
    # 1 Drivable
    # 2 Non-drivable
    # 3 Static obstacle
    # 4 Dynamic object
    # 5 Vegetation
    # 6 Road marking
    # 7 Other

    class_weights = torch.tensor(
        [
            1.0,
            1.5,
            1.2,
            2.0,
            1.5,
            2.0,
            2.0
        ],
        dtype=torch.float32
    ).to(DEVICE)


    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


    # ============================================================
    # TRAINING FUNCTION
    # ============================================================

    def train_one_epoch():

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_points = 0

        progress = tqdm(
            train_loader,
            desc="Training"
        )

        for points, labels in progress:

            points = points.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(points)
                loss = criterion(
                    logits.reshape(-1, NUM_CLASSES),
                    labels.reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += (
                loss.item() *
                labels.numel()
            )

            predictions = torch.argmax(
                logits,
                dim=-1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_points += labels.numel()

            progress.set_postfix(
                loss=loss.item(),
                acc=total_correct / total_points
            )

        return (
            total_loss / total_points,
            total_correct / total_points
        )


    # ============================================================
    # VALIDATION
    # ============================================================

    @torch.no_grad()
    def validate(loader):

        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_points = 0

        class_correct = np.zeros(
            NUM_CLASSES
        )

        class_total = np.zeros(
            NUM_CLASSES
        )

        for points, labels in tqdm(
            loader,
            desc="Validation"
        ):

            points = points.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(points)
                loss = criterion(
                    logits.reshape(
                        -1,
                        NUM_CLASSES
                    ),
                    labels.reshape(-1)
                )

            total_loss += (
                loss.item() *
                labels.numel()
            )

            predictions = torch.argmax(
                logits,
                dim=-1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_points += labels.numel()

            # Per-class accuracy

            for c in range(NUM_CLASSES):

                mask = labels == c

                class_total[c] += mask.sum().item()

                class_correct[c] += (
                    (predictions == c) &
                    mask
                ).sum().item()

        accuracy = (
            total_correct /
            total_points
        )

        class_accuracy = (
            class_correct /
            np.maximum(
                class_total,
                1
            )
        )

        mean_class_accuracy = np.mean(
            class_accuracy
        )

        return (
            total_loss / total_points,
            accuracy,
            mean_class_accuracy,
            class_accuracy
        )


    # ============================================================
    # TRAINING LOOP
    # ============================================================
    def main():

        global best_val_accuracy

        best_val_accuracy = 0.0

        print("\n")
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        for epoch in range(1, EPOCHS + 1):

            print(
                f"\nEpoch {epoch}/{EPOCHS}"
            )

            train_loss, train_acc = (
                train_one_epoch()
            )

            (
                val_loss,
                val_acc,
                mean_class_acc,
                class_acc
            ) = validate(
                val_loader
            )

            scheduler.step()

            print("\nResults:")
            print(
                f"Train Loss: {train_loss:.4f}"
            )

            print(
                f"Train Accuracy: {train_acc * 100:.2f}%"
            )

            print(
                f"Val Loss: {val_loss:.4f}"
            )

            print(
                f"Val Accuracy: {val_acc * 100:.2f}%"
            )

            print(
                f"Mean Class Accuracy: "
                f"{mean_class_acc * 100:.2f}%"
            )

            print("\nClass accuracy:")

            class_names = [
                "Drivable",
                "Non-drivable",
                "Static obstacle",
                "Dynamic object",
                "Vegetation",
                "Road marking",
                "Other"
            ]

            for i, name in enumerate(
                class_names
            ):

                print(
                    f"  {i + 1}: "
                    f"{name:<18} "
                    f"{class_acc[i] * 100:.2f}%"
                )

            # --------------------------------------------------------
            # Save best model
            # --------------------------------------------------------

            if val_acc > best_val_accuracy:

                best_val_accuracy = val_acc

                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR,
                    "pointnet2_best.pth"
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "val_accuracy":
                            val_acc,
                        "mean_class_accuracy":
                            mean_class_acc
                    },
                    checkpoint_path
                )

                print(
                    "\n*** Best model saved ***"
                )

            # --------------------------------------------------------
            # Save latest
            # --------------------------------------------------------

            torch.save(
                model.state_dict(),
                os.path.join(
                    CHECKPOINT_DIR,
                    "pointnet2_latest.pth"
                )
            )

        # ============================================================
        # TEST
        # ============================================================

        print("\n")
        print("=" * 70)
        print("FINAL TEST")
        print("=" * 70)

        checkpoint = torch.load(
            os.path.join(
                CHECKPOINT_DIR,
                "pointnet2_best.pth"
            ),
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        (
            test_loss,
            test_acc,
            test_mean_class_acc,
            test_class_acc
        ) = validate(
            test_loader
        )

        print(
            f"\nTest Loss: {test_loss:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{test_acc * 100:.2f}%"
        )

        print(
            f"Mean Class Accuracy: "
            f"{test_mean_class_acc * 100:.2f}%"
        )

        print("\nTest class accuracy:")

        class_names = [
            "Drivable",
            "Non-drivable",
            "Static obstacle",
            "Dynamic object",
            "Vegetation",
            "Road marking",
            "Other"
        ]

        for i, name in enumerate(
            class_names
        ):

            print(
                f"{i + 1}: "
                f"{name:<18} "
                f"{test_class_acc[i] * 100:.2f}%"
            )

        print("\nTraining complete.")


    if __name__ == "__main__":
        main()