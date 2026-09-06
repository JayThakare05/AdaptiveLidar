import numpy as np


FILE = "frame_000000.npz"


# ============================================================
# LOAD DATASET SAMPLE
# ============================================================

data = np.load(FILE)


points = data["points"]
labels = data["labels"]
frame = data["frame"]


# ============================================================
# BASIC INFORMATION
# ============================================================

print()
print("============================================")
print("POINTNET DATASET SAMPLE")
print("============================================")

print("Frame ID:", int(frame))

print(
    "Points shape:",
    points.shape
)

print(
    "Points dtype:",
    points.dtype
)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Labels dtype:",
    labels.dtype
)


# ============================================================
# POINT INFORMATION
# ============================================================

print()
print("============================================")
print("POINT DATA")
print("============================================")

print(
    "X range:",
    points[:, 0].min(),
    "to",
    points[:, 0].max()
)

print(
    "Y range:",
    points[:, 1].min(),
    "to",
    points[:, 1].max()
)

print(
    "Z range:",
    points[:, 2].min(),
    "to",
    points[:, 2].max()
)

print(
    "Intensity range:",
    points[:, 3].min(),
    "to",
    points[:, 3].max()
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

unique_labels, counts = np.unique(
    labels,
    return_counts=True
)


print()
print("============================================")
print("LABEL DISTRIBUTION")
print("============================================")


for label, count in zip(
    unique_labels,
    counts
):

    percentage = (
        count /
        len(labels)
    ) * 100

    print(
        f"Label {label}: "
        f"{count} points "
        f"({percentage:.2f}%)"
    )


# ============================================================
# CHECK CONSISTENCY
# ============================================================

print()
print("============================================")
print("DATA CONSISTENCY")
print("============================================")

print(
    "Points:",
    len(points)
)

print(
    "Labels:",
    len(labels)
)

print(
    "Point/label match:",
    len(points) == len(labels)
)


# ============================================================
# FIRST 5 POINTS
# ============================================================

print()
print("============================================")
print("FIRST 5 POINTS")
print("============================================")

for i in range(
    min(5, len(points))
):

    print(
        f"{i}: "
        f"{points[i]} "
        f"-> label {labels[i]}"
    )


print()
print("============================================")
print("INSPECTION COMPLETE")
print("============================================")