import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_pointnet2 import PointNet2Segmentation


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = r"..\PointNetDataset"

FRAME_PATH = os.path.join(
    DATASET_ROOT,
    "test",
    "frame_002550.npz"
)

MODEL_PATH = "pointnet2_best.pth"

NUM_POINTS = 2048

NUM_CLASSES = 7


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
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 65)
print("POINTNET++ BEST MODEL - SINGLE FRAME TEST")
print("=" * 65)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(FRAME_PATH):

    raise FileNotFoundError(
        f"Dataset frame not found:\n{FRAME_PATH}"
    )


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print()
print(
    "Loading frame:",
    FRAME_PATH
)

data = np.load(
    FRAME_PATH
)

raw_points = data["points"]

raw_labels = data["labels"]

frame_id = int(
    data["frame"]
)


print(
    "Frame ID:",
    frame_id
)

print(
    "Original points:",
    len(raw_points)
)


# ============================================================
# MATCH JAY'S TRAINING PREPROCESSING
# ============================================================

# XYZ ONLY
points = raw_points[
    :, :3
].astype(
    np.float32
)

labels = raw_labels.astype(
    np.int64
)


# ------------------------------------------------------------
# Labels 1-7 -> 0-6
# ------------------------------------------------------------

labels = (
    labels - 1
)


# ------------------------------------------------------------
# Remove invalid labels
# ------------------------------------------------------------

valid = (
    (labels >= 0)
    &
    (labels < NUM_CLASSES)
)

points = points[
    valid
]

labels = labels[
    valid
]


# ------------------------------------------------------------
# Randomly sample exactly 2048 points
# ------------------------------------------------------------

n = len(points)

rng = np.random.default_rng(
    42
)


if n >= NUM_POINTS:

    indices = rng.choice(
        n,
        NUM_POINTS,
        replace=False
    )

else:

    indices = rng.choice(
        n,
        NUM_POINTS,
        replace=True
    )


points = points[
    indices
]

labels = labels[
    indices
]


# ------------------------------------------------------------
# Normalize exactly like training
# ------------------------------------------------------------

centroid = np.mean(
    points,
    axis=0,
    keepdims=True
)

points = (
    points -
    centroid
)


scale = np.max(
    np.linalg.norm(
        points,
        axis=1
    )
)


if scale > 0:

    points = (
        points /
        scale
    )


print(
    "Model input shape:",
    points.shape
)

print(
    "Normalization scale:",
    float(scale)
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(
    "Creating PointNet++ model..."
)

model = PointNet2Segmentation()

model = model.to(
    DEVICE
)


print(
    "Loading:",
    MODEL_PATH
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


if isinstance(
    checkpoint,
    dict
):

    if (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif (
        "state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# INFERENCE
# ============================================================

input_tensor = (
    torch.from_numpy(
        points
    )
    .unsqueeze(0)
    .to(DEVICE)
)


print()
print(
    "Running inference..."
)


with torch.no_grad():

    logits = model(
        input_tensor
    )

    predictions = torch.argmax(
        logits,
        dim=-1
    )


predictions = (
    predictions
    .squeeze(0)
    .cpu()
    .numpy()
)


print(
    "Inference complete."
)


# ============================================================
# ACCURACY ON THIS SAMPLED SET
# ============================================================

accuracy = np.mean(
    predictions == labels
)


print()
print("=" * 65)
print("FRAME RESULTS")
print("=" * 65)

print(
    f"Sampled points: {len(points)}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print()
print(
    "Predicted classes:"
)


for class_id in range(
    NUM_CLASSES
):

    count = np.sum(
        predictions == class_id
    )

    percentage = (
        count /
        len(predictions)
    ) * 100


    print(
        f"{class_id}: "
        f"{CLASS_NAMES[class_id]:<18} "
        f"{count:>5} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# TOP-DOWN VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7)
)


# ------------------------------------------------------------
# Ground truth
# ------------------------------------------------------------

axes[0].scatter(
    points[:, 0],
    points[:, 1],
    c=labels,
    s=5,
    cmap="tab10",
    vmin=0,
    vmax=6
)

axes[0].set_title(
    "Ground Truth"
)

axes[0].set_xlabel(
    "X"
)

axes[0].set_ylabel(
    "Y"
)

axes[0].axis(
    "equal"
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

axes[1].scatter(
    points[:, 0],
    points[:, 1],
    c=predictions,
    s=5,
    cmap="tab10",
    vmin=0,
    vmax=6
)

axes[1].set_title(
    "PointNet++ Prediction"
)

axes[1].set_xlabel(
    "X"
)

axes[1].set_ylabel(
    "Y"
)

axes[1].axis(
    "equal"
)


plt.suptitle(
    f"PointNet++ Frame {frame_id}"
)

plt.tight_layout()


output = (
    f"pointnet_best_frame_"
    f"{frame_id}.png"
)


plt.savefig(
    output,
    dpi=200
)

plt.show()


print()
print(
    "Saved visualization:",
    output
)

print()
print(
    "=" * 65
)

print(
    "SINGLE-FRAME MODEL TEST COMPLETE"
)

print(
    "=" * 65
)