import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os


# ============================================================
# CONFIGURATION
# ============================================================

NPZ_FILE = "frame_000000.npz"

GRID_SIZE = 0.5


# ============================================================
# SEMANTIC CLASSES
# ============================================================

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
# LOAD DATASET
# ============================================================

print(
    "Loading:",
    os.path.abspath(NPZ_FILE)
)


if not os.path.exists(NPZ_FILE):

    print(
        "ERROR: Dataset file not found."
    )

    exit()


data = np.load(
    NPZ_FILE
)


points = data["points"]

labels = data["labels"]

frame_id = int(
    data["frame"]
)


# ============================================================
# VALIDATION
# ============================================================

print()
print(
    "============================================"
)

print(
    "SEMANTIC 2.5D MAPPING"
)

print(
    "============================================"
)

print(
    "Frame:",
    frame_id
)

print(
    "Points:",
    len(points)
)

print(
    "Labels:",
    len(labels)
)


if len(points) != len(labels):

    print(
        "ERROR: Number of points and labels do not match."
    )

    exit()


# ============================================================
# EXTRACT DATA
# ============================================================

x = points[:, 0]

y = points[:, 1]

z = points[:, 2]

intensity = points[:, 3]


# ============================================================
# FILTER INVALID POINTS
# ============================================================

valid = (
    np.isfinite(x) &
    np.isfinite(y) &
    np.isfinite(z) &
    np.isfinite(intensity) &
    np.isin(labels, list(CLASS_NAMES.keys()))
)


x = x[valid]

y = y[valid]

z = z[valid]

intensity = intensity[valid]

labels = labels[valid]


print()
print(
    "Valid points:",
    len(x)
)


# ============================================================
# GRID BOUNDS
# ============================================================

x_min = x.min()

x_max = x.max()

y_min = y.min()

y_max = y.max()


grid_width = (
    int(
        np.ceil(
            (x_max - x_min)
            / GRID_SIZE
        )
    )
    + 1
)

grid_height = (
    int(
        np.ceil(
            (y_max - y_min)
            / GRID_SIZE
        )
    )
    + 1
)


print()
print(
    "Grid resolution:",
    GRID_SIZE,
    "m"
)

print(
    "Grid size:",
    grid_width,
    "x",
    grid_height
)


# ============================================================
# GRID INDICES
# ============================================================

grid_x = (
    (x - x_min)
    / GRID_SIZE
).astype(np.int32)


grid_y = (
    (y - y_min)
    / GRID_SIZE
).astype(np.int32)


# ============================================================
# MAP STORAGE
# ============================================================

height_map = np.full(
    (
        grid_height,
        grid_width
    ),
    np.nan,
    dtype=np.float32
)


semantic_map = np.full(
    (
        grid_height,
        grid_width
    ),
    -1,
    dtype=np.int8
)


point_count_map = np.zeros(
    (
        grid_height,
        grid_width
    ),
    dtype=np.int32
)


# ============================================================
# STORE LABELS PER CELL
# ============================================================

cell_labels = {}

cell_z = {}


for i in range(len(z)):

    gx = grid_x[i]

    gy = grid_y[i]

    key = (
        gx,
        gy
    )


    if key not in cell_labels:

        cell_labels[key] = []

        cell_z[key] = []


    cell_labels[key].append(
        int(labels[i])
    )

    cell_z[key].append(
        z[i]
    )


# ============================================================
# BUILD 2.5D MAP
# ============================================================

for (
    gx,
    gy
), label_values in cell_labels.items():

    label_values = np.asarray(
        label_values
    )

    z_values = np.asarray(
        cell_z[
            (gx, gy)
        ]
    )


    # --------------------------------------------------------
    # HEIGHT
    # --------------------------------------------------------

    # Use 95th percentile instead of a raw maximum
    # to reduce the effect of extreme returns.

    cell_height = np.percentile(
        z_values,
        95
    )


    height_map[
        gy,
        gx
    ] = cell_height


    # --------------------------------------------------------
    # SEMANTIC CLASS
    # --------------------------------------------------------

    unique_labels, counts = np.unique(
        label_values,
        return_counts=True
    )


    majority_index = np.argmax(
        counts
    )


    majority_label = (
        unique_labels[
            majority_index
        ]
    )


    semantic_map[
        gy,
        gx
    ] = majority_label


    # --------------------------------------------------------
    # POINT COUNT
    # --------------------------------------------------------

    point_count_map[
        gy,
        gx
    ] = len(label_values)


# ============================================================
# STATISTICS
# ============================================================

observed_cells = np.sum(
    semantic_map >= 0
)

total_cells = (
    grid_width *
    grid_height
)


coverage = (
    observed_cells /
    total_cells
) * 100


print()
print(
    "============================================"
)

print(
    "MAP RESULTS"
)

print(
    "============================================"
)

print(
    "Observed cells:",
    observed_cells
)

print(
    "Total cells:",
    total_cells
)

print(
    "Coverage:",
    round(
        coverage,
        2
    ),
    "%"
)


# ============================================================
# CLASS COUNTS
# ============================================================

print()
print(
    "Semantic cell distribution:"
)


for label_id in sorted(
    CLASS_NAMES.keys()
):

    count = np.sum(
        semantic_map == label_id
    )

    if count > 0:

        percentage = (
            count /
            observed_cells
        ) * 100

        print(
            f"{label_id}: "
            f"{CLASS_NAMES[label_id]} "
            f"→ "
            f"{count} cells "
            f"({percentage:.2f}%)"
        )


# ============================================================
# SAVE ARRAYS
# ============================================================

np.save(
    "semantic_2_5d_height.npy",
    height_map
)

np.save(
    "semantic_2_5d_labels.npy",
    semantic_map
)

np.save(
    "semantic_2_5d_point_count.npy",
    point_count_map
)


# ============================================================
# SEMANTIC VISUALIZATION
# ============================================================

# Colors are only used for semantic visualization.

colors = [
    "#808080",  # Drivable
    "#D2D2D2",  # Non-drivable
    "#FF3030",  # Static obstacle
    "#FF8C00",  # Dynamic object
    "#228B22",  # Vegetation
    "#FFD700",  # Road marking
    "#800080"   # Other
]


cmap = ListedColormap(
    colors
)


display_map = (
    semantic_map.astype(
        float
    )
)

display_map[
    semantic_map < 0
] = np.nan


plt.figure(
    figsize=(12, 9)
)


plt.imshow(
    display_map,
    origin="lower",
    interpolation="nearest",
    cmap=cmap,
    vmin=0,
    vmax=6
)


plt.title(
    "Semantic 2.5D LiDAR Map"
)


plt.xlabel(
    "X Grid Cell"
)


plt.ylabel(
    "Y Grid Cell"
)


legend_items = []

for label_id, name in CLASS_NAMES.items():

    legend_items.append(
        Patch(
            facecolor=colors[label_id],
            label=(
                f"{label_id} - {name}"
            )
        )
    )


plt.legend(
    handles=legend_items,
    bbox_to_anchor=(
        1.05,
        1
    ),
    loc="upper left"
)


plt.tight_layout()


plt.savefig(
    "semantic_2_5d_map.png",
    dpi=200,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# HEIGHT MAP
# ============================================================

plt.figure(
    figsize=(12, 9)
)


height_display = (
    height_map.copy()
)


plt.imshow(
    height_display,
    origin="lower",
    interpolation="nearest",
    cmap="terrain"
)


plt.title(
    "Semantic 2.5D Height Map"
)


plt.xlabel(
    "X Grid Cell"
)


plt.ylabel(
    "Y Grid Cell"
)


plt.colorbar(
    label="Height (m)"
)


plt.tight_layout()


plt.savefig(
    "semantic_2_5d_height_map.png",
    dpi=200
)


plt.show()


# ============================================================
# COMPLETE
# ============================================================

print()
print(
    "============================================"
)

print(
    "SEMANTIC 2.5D MAPPING COMPLETE"
)

print(
    "============================================"
)

print(
    "Saved:"
)

print(
    "semantic_2_5d_height.npy"
)

print(
    "semantic_2_5d_labels.npy"
)

print(
    "semantic_2_5d_point_count.npy"
)

print(
    "semantic_2_5d_map.png"
)

print(
    "semantic_2_5d_height_map.png"
)