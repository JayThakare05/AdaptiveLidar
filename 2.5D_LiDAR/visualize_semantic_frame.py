import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"..\PointNetDataset"

SPLIT = "train"

FRAME_INDEX = 1000


CLASS_NAMES = {
    0: "Drivable",
    1: "Non-drivable",
    2: "Static obstacle",
    3: "Dynamic object",
    4: "Vegetation",
    5: "Road marking",
    6: "Other",
    7: "Unknown"
}


# ============================================================
# COLORS FOR VISUALIZATION
# ============================================================

CLASS_COLORS = {
    0: "gray",
    1: "lightgray",
    2: "red",
    3: "orange",
    4: "green",
    5: "gold",
    6: "purple",
    7: "black"
}


# ============================================================
# FIND FILES
# ============================================================

split_path = os.path.join(
    DATASET_ROOT,
    SPLIT
)

files = sorted(
    glob.glob(
        os.path.join(
            split_path,
            "*.npz"
        )
    )
)


if len(files) == 0:

    print(
        "ERROR: No .npz files found."
    )

    exit()


if FRAME_INDEX >= len(files):

    print(
        "ERROR: FRAME_INDEX is outside dataset."
    )

    exit()


file_path = files[
    FRAME_INDEX
]


# ============================================================
# LOAD
# ============================================================

print(
    "Loading:",
    os.path.abspath(file_path)
)


data = np.load(
    file_path
)

points = data["points"]

labels = data["labels"]

frame_id = int(
    data["frame"]
)


print()
print(
    "============================================"
)

print(
    "SEMANTIC LiDAR VISUALIZATION"
)

print(
    "============================================"
)

print(
    "Split:",
    SPLIT
)

print(
    "File:",
    os.path.basename(file_path)
)

print(
    "Frame ID:",
    frame_id
)

print(
    "Points:",
    len(points)
)


# ============================================================
# TOP-DOWN COORDINATES
# ============================================================

x = points[:, 0]

y = points[:, 1]


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(12, 10)
)


legend_handles = []


for label_id in sorted(
    np.unique(labels)
):

    mask = (
        labels == label_id
    )

    name = CLASS_NAMES.get(
        int(label_id),
        "Unknown"
    )

    color = CLASS_COLORS.get(
        int(label_id),
        "black"
    )


    plt.scatter(
        x[mask],
        y[mask],
        s=2,
        c=color,
        label=(
            f"{label_id} - {name}"
        )
    )


plt.title(
    f"Semantic LiDAR Frame {frame_id}"
)

plt.xlabel(
    "X (m)"
)

plt.ylabel(
    "Y (m)"
)

plt.axis(
    "equal"
)

plt.legend(
    loc="upper right"
)


plt.tight_layout()


output = (
    f"semantic_frame_"
    f"{frame_id}.png"
)


plt.savefig(
    output,
    dpi=200
)

plt.show()


print()
print(
    "Saved:",
    os.path.abspath(output)
)


# ============================================================
# PER-FRAME LABEL COUNTS
# ============================================================

print()
print(
    "Label distribution:"
)


for label_id in sorted(
    np.unique(labels)
):

    count = np.sum(
        labels == label_id
    )

    percentage = (
        count /
        len(labels)
    ) * 100


    print(
        f"{label_id}: "
        f"{CLASS_NAMES.get(label_id, 'Unknown'):<18} "
        f"{count:>6} "
        f"({percentage:>6.2f}%)"
    )


print()
print(
    "Visualization complete."
)