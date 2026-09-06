import os
import glob
import numpy as np
from collections import Counter
import time


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"..\PointNetDataset"

SPLITS = [
    "train",
    "val",
    "test"
]


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
# HELPERS
# ============================================================

def format_number(number):
    return f"{number:,}"


def format_mb(value):
    return f"{value / (1024 * 1024):.2f} MB"


# ============================================================
# START
# ============================================================

print()
print("============================================================")
print("POINTNET DATASET ANALYSIS")
print("============================================================")

print(
    "Dataset root:",
    os.path.abspath(DATASET_ROOT)
)

print()


overall_start = time.perf_counter()


# ============================================================
# GLOBAL STATISTICS
# ============================================================

total_frames = 0
total_points = 0

global_label_counts = Counter()

split_stats = {}


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

for split in SPLITS:

    split_path = os.path.join(
        DATASET_ROOT,
        split
    )

    files = sorted(
        glob.glob(
            os.path.join(
                split_path,
                "*.npz"
            )
        )
    )

    print()
    print("------------------------------------------------------------")
    print(f"SPLIT: {split.upper()}")
    print("------------------------------------------------------------")

    print(
        "Path:",
        os.path.abspath(split_path)
    )

    print(
        "Files:",
        len(files)
    )


    split_point_count = 0
    split_label_counts = Counter()

    min_points = None
    max_points = None

    x_min = np.inf
    x_max = -np.inf

    y_min = np.inf
    y_max = -np.inf

    z_min = np.inf
    z_max = -np.inf


    # --------------------------------------------------------
    # PROCESS FILES
    # --------------------------------------------------------

    for index, file_path in enumerate(files):

        try:

            data = np.load(
                file_path
            )

            points = data["points"]
            labels = data["labels"]


            # ------------------------------------------------
            # BASIC VALIDATION
            # ------------------------------------------------

            if len(points) != len(labels):

                print(
                    "WARNING:",
                    os.path.basename(file_path),
                    "has mismatched points/labels."
                )

                continue


            # ------------------------------------------------
            # FRAME STATISTICS
            # ------------------------------------------------

            number_of_points = len(points)

            split_point_count += (
                number_of_points
            )

            total_frames += 1
            total_points += number_of_points


            # ------------------------------------------------
            # LABEL COUNTS
            # ------------------------------------------------

            unique, counts = np.unique(
                labels,
                return_counts=True
            )

            for label, count in zip(
                unique,
                counts
            ):

                label = int(label)
                count = int(count)

                split_label_counts[
                    label
                ] += count

                global_label_counts[
                    label
                ] += count


            # ------------------------------------------------
            # POINT COUNT RANGE
            # ------------------------------------------------

            if (
                min_points is None
                or number_of_points < min_points
            ):

                min_points = number_of_points


            if (
                max_points is None
                or number_of_points > max_points
            ):

                max_points = number_of_points


            # ------------------------------------------------
            # XYZ RANGES
            # ------------------------------------------------

            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]


            x_min = min(
                x_min,
                float(np.min(x))
            )

            x_max = max(
                x_max,
                float(np.max(x))
            )

            y_min = min(
                y_min,
                float(np.min(y))
            )

            y_max = max(
                y_max,
                float(np.max(y))
            )

            z_min = min(
                z_min,
                float(np.min(z))
            )

            z_max = max(
                z_max,
                float(np.max(z))
            )


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (
                (index + 1) % 100 == 0
                or
                index == len(files) - 1
            ):

                print(
                    f"\rProcessed: "
                    f"{index + 1}/{len(files)}",
                    end=""
                )


        except Exception as e:

            print()
            print(
                "ERROR reading:",
                file_path
            )

            print(
                "Reason:",
                e
            )


    print()


    # ========================================================
    # SPLIT RESULTS
    # ========================================================

    split_stats[
        split
    ] = {
        "frames": len(files),
        "points": split_point_count,
        "labels": split_label_counts,
        "min_points": min_points,
        "max_points": max_points,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max
    }


    print()
    print(
        "Total points:",
        format_number(
            split_point_count
        )
    )

    print(
        "Points/frame:",
        (
            f"{min_points:,}"
            if min_points is not None
            else "N/A"
        ),
        "to",
        (
            f"{max_points:,}"
            if max_points is not None
            else "N/A"
        )
    )

    print()
    print("Label distribution:")


    for label_id in sorted(
        split_label_counts.keys()
    ):

        count = split_label_counts[
            label_id
        ]

        percentage = (
            count /
            split_point_count
        ) * 100


        name = CLASS_NAMES.get(
            label_id,
            "Unknown"
        )


        print(
            f"  {label_id}: "
            f"{name:<18} "
            f"{count:>12,} "
            f"({percentage:>6.2f}%)"
        )


    print()
    print("XYZ range:")

    print(
        f"  X: {x_min:.3f} "
        f"to {x_max:.3f}"
    )

    print(
        f"  Y: {y_min:.3f} "
        f"to {y_max:.3f}"
    )

    print(
        f"  Z: {z_min:.3f} "
        f"to {z_max:.3f}"
    )


# ============================================================
# GLOBAL RESULTS
# ============================================================

elapsed = (
    time.perf_counter()
    -
    overall_start
)


print()
print()
print("============================================================")
print("GLOBAL DATASET SUMMARY")
print("============================================================")

print(
    "Frames:",
    format_number(
        total_frames
    )
)

print(
    "Total points:",
    format_number(
        total_points
    )
)

print(
    "Analysis time:",
    f"{elapsed:.2f} seconds"
)


# ============================================================
# GLOBAL LABEL DISTRIBUTION
# ============================================================

print()
print(
    "GLOBAL LABEL DISTRIBUTION"
)

print(
    "------------------------------------------------------------"
)


for label_id in sorted(
    CLASS_NAMES.keys()
):

    count = global_label_counts[
        label_id
    ]

    percentage = (
        count /
        total_points
    ) * 100


    print(
        f"{label_id}: "
        f"{CLASS_NAMES[label_id]:<18} "
        f"{count:>15,} "
        f"({percentage:>6.2f}%)"
    )


# ============================================================
# CHECK FOR UNKNOWN LABELS
# ============================================================

known_labels = set(
    CLASS_NAMES.keys()
)

unknown_labels = (
    set(global_label_counts.keys())
    -
    known_labels
)


print()

if unknown_labels:

    print(
        "WARNING: Unknown labels found:",
        sorted(unknown_labels)
    )

else:

    print(
        "All labels are within expected "
        "0-6 class range."
    )


# ============================================================
# SPLIT SUMMARY
# ============================================================

print()
print(
    "============================================================"
)

print(
    "SPLIT SUMMARY"
)

print(
    "============================================================"
)


for split in SPLITS:

    stats = split_stats[
        split
    ]

    print()

    print(
        f"{split.upper()}:"
    )

    print(
        "  Frames:",
        format_number(
            stats["frames"]
        )
    )

    print(
        "  Points:",
        format_number(
            stats["points"]
        )
    )


# ============================================================
# FINAL
# ============================================================

print()
print(
    "============================================================"
)

print(
    "DATASET ANALYSIS COMPLETE"
)

print(
    "============================================================"
)