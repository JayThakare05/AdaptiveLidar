import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# ============================================================
# IMPORT THE CORRECT MODEL CLASS
# ============================================================

from train_pointnet2 import PointNet2Segmentation


# ============================================================
# SETTINGS
# ============================================================

TEST_DIR = r"..\PointNetDataset\test"

MODEL_PATH = "pointnet2_best.pth"

NUM_CLASSES = 7

# Must match NUM_POINTS used in train_pointnet2.py
NUM_POINTS = 2048

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = {
    0: "Drivable",
    1: "Non-drivable",
    2: "Static obstacle",
    3: "Dynamic object",
    4: "Vegetation",
    5: "Road marking",
    6: "Other"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("POINTNET++ FINAL TESTING")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(self, directory):

        self.files = sorted(
            glob.glob(
                os.path.join(
                    directory,
                    "*.npz"
                )
            )
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No NPZ files found in {directory}"
            )

        print(
            f"Loaded {len(self.files)} test frames"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        path = self.files[index]

        data = np.load(path)

        # ----------------------------------------------------
        # IMPORTANT:
        # Use XYZ only.
        # Do NOT pass intensity.
        # ----------------------------------------------------

        points = data["points"][:, :3].astype(
            np.float32
        )

        labels = data["labels"].astype(
            np.int64
        )

        # ----------------------------------------------------
        # FIX #1: Convert labels 1-7 -> 0-6, EXACTLY like training.
        # Raw labels in the npz are 1..7 (verified: 0 never appears).
        # The model was trained to predict 0..6, so ground-truth
        # labels must be shifted the same way here or every
        # comparison in the confusion matrix is off by one class.
        # ----------------------------------------------------

        labels = labels - 1

        # ----------------------------------------------------
        # Remove invalid labels (mirrors training)
        # ----------------------------------------------------

        valid = (
            (labels >= 0) &
            (labels < NUM_CLASSES)
        )

        points = points[valid]
        labels = labels[valid]

        # ----------------------------------------------------
        # FIX #2: Sample to a fixed NUM_POINTS, EXACTLY like
        # training (same replace=False/True logic). Feeding the
        # full, variable-length frame is a train/test mismatch.
        # ----------------------------------------------------

        n = len(points)

        if n >= NUM_POINTS:
            indices = np.random.choice(
                n, NUM_POINTS, replace=False
            )
        else:
            indices = np.random.choice(
                n, NUM_POINTS, replace=True
            )

        points = points[indices]
        labels = labels[indices]

        # ----------------------------------------------------
        # FIX #3: Normalize xyz EXACTLY like training
        # (center on centroid, scale to unit sphere). The model
        # has only ever seen normalized coordinates - raw world
        # coordinates (e.g. x ~ -76) are far outside anything it
        # learned to handle.
        # ----------------------------------------------------

        centroid = np.mean(points, axis=0, keepdims=True)
        points = points - centroid

        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale

        return (
            torch.from_numpy(points),
            torch.from_numpy(labels)
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nCreating model...")

model = PointNet2Segmentation()

model = model.to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print(
    f"\nLoading model: {MODEL_PATH}"
)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Checkpoint not found:\n{MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


# Support different checkpoint formats

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

test_dataset = TestDataset(
    TEST_DIR
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("RUNNING FINAL TEST")
print("=" * 70)

total_frames = len(test_dataset)

with torch.no_grad():

    for i in tqdm(
        range(total_frames),
        desc="Testing"
    ):

        points, labels = test_dataset[i]

        # Add batch dimension
        points = points.unsqueeze(0)

        points = points.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        logits = model(points)

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        predictions = (
            predictions
            .reshape(-1)
            .cpu()
            .numpy()
        )

        labels_np = (
            labels
            .reshape(-1)
            .numpy()
        )

        # ----------------------------------------------------
        # VALID LABELS
        # ----------------------------------------------------

        valid = (
            (labels_np >= 0)
            &
            (labels_np < NUM_CLASSES)
        )

        true_labels = labels_np[valid]

        pred_labels = predictions[valid]

        # ----------------------------------------------------
        # UPDATE CONFUSION MATRIX
        # ----------------------------------------------------

        np.add.at(
            confusion,
            (true_labels, pred_labels),
            1
        )


# ============================================================
# METRICS
# ============================================================

total = confusion.sum()

correct = np.trace(confusion)

accuracy = (
    correct / total
    if total > 0
    else 0.0
)


ious = []
precisions = []
recalls = []


for cls in range(NUM_CLASSES):

    tp = confusion[cls, cls]

    fp = (
        confusion[:, cls].sum()
        - tp
    )

    fn = (
        confusion[cls, :].sum()
        - tp
    )

    union = tp + fp + fn

    # IoU
    if union > 0:
        iou = tp / union
    else:
        iou = np.nan

    # Precision
    if tp + fp > 0:
        precision = tp / (
            tp + fp
        )
    else:
        precision = np.nan

    # Recall
    if tp + fn > 0:
        recall = tp / (
            tp + fn
        )
    else:
        recall = np.nan

    ious.append(iou)
    precisions.append(precision)
    recalls.append(recall)


miou = np.nanmean(ious)

mean_precision = np.nanmean(
    precisions
)

mean_recall = np.nanmean(
    recalls
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("POINTNET++ FINAL TEST RESULTS")
print("=" * 70)

print(
    f"\nTest frames: {total_frames}"
)

print(
    f"Overall Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Mean IoU: "
    f"{miou * 100:.2f}%"
)

print(
    f"Mean Precision: "
    f"{mean_precision * 100:.2f}%"
)

print(
    f"Mean Recall: "
    f"{mean_recall * 100:.2f}%"
)


# ============================================================
# PER CLASS
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

print(
    f"{'Class':<7}"
    f"{'Name':<22}"
    f"{'IoU':>10}"
    f"{'Precision':>14}"
    f"{'Recall':>12}"
)

print("-" * 70)


for cls in range(NUM_CLASSES):

    iou = ious[cls]

    precision = precisions[cls]

    recall = recalls[cls]

    iou_text = (
        f"{iou * 100:.2f}%"
        if not np.isnan(iou)
        else "N/A"
    )

    precision_text = (
        f"{precision * 100:.2f}%"
        if not np.isnan(precision)
        else "N/A"
    )

    recall_text = (
        f"{recall * 100:.2f}%"
        if not np.isnan(recall)
        else "N/A"
    )

    print(
        f"{cls:<7}"
        f"{CLASS_NAMES[cls]:<22}"
        f"{iou_text:>10}"
        f"{precision_text:>14}"
        f"{recall_text:>12}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nRows = Ground Truth")
print("Columns = Prediction\n")

print(confusion)


# ============================================================
# PREDICTED CLASS DISTRIBUTION
# ============================================================

predicted_distribution = confusion.sum(
    axis=0
)

print("\n" + "=" * 70)
print("PREDICTED CLASS DISTRIBUTION")
print("=" * 70)

for cls in range(NUM_CLASSES):

    count = predicted_distribution[cls]

    percentage = (
        count / total * 100
        if total > 0
        else 0
    )

    print(
        f"{cls}: "
        f"{CLASS_NAMES[cls]:<20}"
        f"{count:>12,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "test_results",
    exist_ok=True
)

result_file = os.path.join(
    "test_results",
    "final_test_results.txt"
)


with open(
    result_file,
    "w"
) as f:

    f.write(
        "POINTNET++ FINAL TEST RESULTS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Test frames: {total_frames}\n"
    )

    f.write(
        f"Overall Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Mean IoU: "
        f"{miou * 100:.2f}%\n"
    )

    f.write(
        f"Mean Precision: "
        f"{mean_precision * 100:.2f}%\n"
    )

    f.write(
        f"Mean Recall: "
        f"{mean_recall * 100:.2f}%\n\n"
    )

    f.write(
        "PER-CLASS RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for cls in range(NUM_CLASSES):

        f.write(
            f"{cls}: "
            f"{CLASS_NAMES[cls]}\n"
        )

        f.write(
            f"  IoU: "
            f"{ious[cls] * 100:.2f}%\n"
        )

        f.write(
            f"  Precision: "
            f"{precisions[cls] * 100:.2f}%\n"
        )

        f.write(
            f"  Recall: "
            f"{recalls[cls] * 100:.2f}%\n"
        )

    f.write(
        "\nCONFUSION MATRIX\n"
    )

    f.write(
        str(confusion)
    )

    f.write(
        "\n\nPREDICTED CLASS DISTRIBUTION\n"
    )

    for cls in range(NUM_CLASSES):

        f.write(
            f"{cls}: "
            f"{CLASS_NAMES[cls]} "
            f"{predicted_distribution[cls]:,}\n"
        )


print("\n" + "=" * 70)
print("FINAL TEST COMPLETE")
print("=" * 70)

print(
    f"\nResults saved to:"
    f"\n{result_file}"
)