import os
import glob
import csv
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "carla_pointnet_predictions"

OUTPUT_DIR = "semantic_adaptive_2_5d"

MAX_FRAMES = 200


# ============================================================
# ADAPTIVE RESOLUTION
# ============================================================

BASE_CELL_SIZE = 2.0
MEDIUM_CELL_SIZE = 1.0
FINE_CELL_SIZE = 0.5


# Classes that deserve highest spatial resolution.
FINE_CLASSES = {
    2,   # Static obstacle
    3,   # Dynamic object
    4    # Vegetation
}


# Medium resolution.
MEDIUM_CLASSES = {
    1,   # Non-drivable
    5    # Road marking
}


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
# FIXED COLORS
# ============================================================

CLASS_COLORS = {
    0: "#707070",
    1: "#BDBDBD",
    2: "#E53935",
    3: "#FB8C00",
    4: "#43A047",
    5: "#FDD835",
    6: "#8E44AD"
}


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


FRAME_DIR = os.path.join(
    OUTPUT_DIR,
    "frames"
)


os.makedirs(
    FRAME_DIR,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

prediction_files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "prediction_*.npz"
        )
    )
)


print()
print("=" * 70)
print("SEMANTIC ADAPTIVE 2.5D MAPPING")
print("=" * 70)

print(
    "Prediction directory:",
    os.path.abspath(INPUT_DIR)
)

print(
    "Prediction files found:",
    len(prediction_files)
)


if len(prediction_files) == 0:

    raise RuntimeError(
        "\nNo prediction files found.\n\n"
        "First modify carla_pointnet_offline_inference.py "
        "to save prediction_XXXXXX.npz files, then rerun "
        "the offline inference."
    )


prediction_files = (
    prediction_files[:MAX_FRAMES]
)


print(
    "Frames selected:",
    len(prediction_files)
)


# ============================================================
# CREATE ADAPTIVE CELLS
# ============================================================

def create_adaptive_cells(
    points,
    predictions
):

    if len(points) == 0:

        return []


    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]


    # --------------------------------------------------------
    # Find base-grid origin
    # --------------------------------------------------------

    x_min = (
        math.floor(
            float(np.min(x))
            /
            BASE_CELL_SIZE
        )
        *
        BASE_CELL_SIZE
    )


    y_min = (
        math.floor(
            float(np.min(y))
            /
            BASE_CELL_SIZE
        )
        *
        BASE_CELL_SIZE
    )


    # --------------------------------------------------------
    # Base grid indices
    # --------------------------------------------------------

    bx = np.floor(
        (
            x -
            x_min
        )
        /
        BASE_CELL_SIZE
    ).astype(
        np.int32
    )


    by = np.floor(
        (
            y -
            y_min
        )
        /
        BASE_CELL_SIZE
    ).astype(
        np.int32
    )


    base_keys = np.column_stack(
        [
            bx,
            by
        ]
    )


    unique_base = np.unique(
        base_keys,
        axis=0
    )


    cells = []


    # ========================================================
    # PROCESS BASE CELLS
    # ========================================================

    for base_key in unique_base:

        base_x_index = int(
            base_key[0]
        )

        base_y_index = int(
            base_key[1]
        )


        base_mask = (
            (
                bx ==
                base_x_index
            )
            &
            (
                by ==
                base_y_index
            )
        )


        base_points = points[
            base_mask
        ]


        base_predictions = predictions[
            base_mask
        ]


        if len(base_points) == 0:

            continue


        # ----------------------------------------------------
        # Determine dominant semantic class
        # ----------------------------------------------------

        class_counts = np.bincount(
            base_predictions,
            minlength=7
        )


        dominant_class = int(
            np.argmax(
                class_counts
            )
        )


        # ----------------------------------------------------
        # Determine adaptive resolution
        # ----------------------------------------------------

        if (
            dominant_class
            in
            FINE_CLASSES
        ):

            cell_size = (
                FINE_CELL_SIZE
            )

        elif (
            dominant_class
            in
            MEDIUM_CLASSES
        ):

            cell_size = (
                MEDIUM_CELL_SIZE
            )

        else:

            cell_size = (
                BASE_CELL_SIZE
            )


        # ====================================================
        # BASE CELL
        # ====================================================

        if (
            cell_size ==
            BASE_CELL_SIZE
        ):

            cell_x = (
                x_min
                +
                base_x_index *
                BASE_CELL_SIZE
            )


            cell_y = (
                y_min
                +
                base_y_index *
                BASE_CELL_SIZE
            )


            # Use median height rather than maximum.
            # This reduces the effect of isolated
            # high points/outliers.

            cell_height = float(
                np.median(
                    base_points[:, 2]
                )
            )


            cells.append(
                {
                    "x":
                        cell_x,

                    "y":
                        cell_y,

                    "size":
                        BASE_CELL_SIZE,

                    "class":
                        dominant_class,

                    "height":
                        cell_height,

                    "points":
                        len(base_points)
                }
            )


            continue


        # ====================================================
        # REFINED CELL
        # ====================================================

        origin_x = (
            x_min
            +
            base_x_index *
            BASE_CELL_SIZE
        )


        origin_y = (
            y_min
            +
            base_y_index *
            BASE_CELL_SIZE
        )


        sub_x = np.floor(
            (
                base_points[:, 0]
                -
                origin_x
            )
            /
            cell_size
        ).astype(
            np.int32
        )


        sub_y = np.floor(
            (
                base_points[:, 1]
                -
                origin_y
            )
            /
            cell_size
        ).astype(
            np.int32
        )


        sub_keys = np.column_stack(
            [
                sub_x,
                sub_y
            ]
        )


        unique_sub = np.unique(
            sub_keys,
            axis=0
        )


        for sub_key in unique_sub:

            sx = int(
                sub_key[0]
            )

            sy = int(
                sub_key[1]
            )


            sub_mask = (
                (
                    sub_x ==
                    sx
                )
                &
                (
                    sub_y ==
                    sy
                )
            )


            sub_points = (
                base_points[
                    sub_mask
                ]
            )


            sub_predictions = (
                base_predictions[
                    sub_mask
                ]
            )


            if len(sub_points) == 0:

                continue


            sub_counts = np.bincount(
                sub_predictions,
                minlength=7
            )


            sub_class = int(
                np.argmax(
                    sub_counts
                )
            )


            cell_x = (
                origin_x
                +
                sx *
                cell_size
            )


            cell_y = (
                origin_y
                +
                sy *
                cell_size
            )


            cell_height = float(
                np.median(
                    sub_points[:, 2]
                )
            )


            cells.append(
                {
                    "x":
                        cell_x,

                    "y":
                        cell_y,

                    "size":
                        cell_size,

                    "class":
                        sub_class,

                    "height":
                        cell_height,

                    "points":
                        len(sub_points)
                }
            )


    return cells


# ============================================================
# DRAW MAP
# ============================================================

def draw_map(
    cells,
    frame_id,
    output_path,
    show_height=False
):

    if len(cells) == 0:

        return


    fig, ax = plt.subplots(
        figsize=(14, 9)
    )


    # --------------------------------------------------------
    # Draw cells
    # --------------------------------------------------------

    for cell in cells:

        class_id = cell[
            "class"
        ]

        size = cell[
            "size"
        ]


        rect = Rectangle(

            (
                cell["x"],
                cell["y"]
            ),

            size,
            size,

            facecolor=(
                CLASS_COLORS[
                    class_id
                ]
            ),

            edgecolor="white",

            linewidth=0.25,

            alpha=0.88
        )


        ax.add_patch(
            rect
        )


    # --------------------------------------------------------
    # Find dynamic and vegetation cells
    # --------------------------------------------------------

    dynamic_cells = [
        c
        for c in cells
        if c["class"] == 3
    ]


    vegetation_cells = [
        c
        for c in cells
        if c["class"] == 4
    ]


    static_cells = [
        c
        for c in cells
        if c["class"] == 2
    ]


    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    ax.set_aspect(
        "equal",
        adjustable="box"
    )


    ax.set_xlabel(
        "Local X (m)"
    )


    ax.set_ylabel(
        "Local Y (m)"
    )


    ax.set_title(
        "Semantic Adaptive 2.5D LiDAR Map\n"
        f"CARLA Frame {frame_id} | "
        f"Adaptive Cells: {len(cells)}"
    )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    handles = []


    for class_id in range(
        7
    ):

        handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=(
                    CLASS_COLORS[
                        class_id
                    ]
                ),
                edgecolor="black",
                label=(
                    f"{class_id} - "
                    f"{CLASS_NAMES[class_id]}"
                )
            )
        )


    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8
    )


    # --------------------------------------------------------
    # Statistics box
    # --------------------------------------------------------

    ax.text(

        0.02,
        0.02,

        (
            f"Dynamic cells: "
            f"{len(dynamic_cells)}\n"
            f"Vegetation cells: "
            f"{len(vegetation_cells)}\n"
            f"Static obstacle cells: "
            f"{len(static_cells)}\n"
            f"Resolution: "
            f"2.0m / 1.0m / 0.5m"
        ),

        transform=ax.transAxes,

        fontsize=8,

        verticalalignment="bottom",

        bbox={
            "boxstyle":
                "round",

            "facecolor":
                "white",

            "alpha":
                0.85
        }
    )


    # --------------------------------------------------------
    # Optional height annotation
    # --------------------------------------------------------

    if show_height:

        for cell in cells:

            if cell[
                "points"
            ] < 5:

                continue


            if cell[
                "class"
            ] not in {
                2,
                3,
                4
            }:

                continue


            ax.text(

                cell["x"] +
                cell["size"] /
                2,

                cell["y"] +
                cell["size"] /
                2,

                f"{cell['height']:.1f}",

                ha="center",

                va="center",

                fontsize=4,

                color="black"
            )


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# ============================================================
# PROCESS ALL PREDICTION FILES
# ============================================================

summary_rows = []

best_frame = None

best_dynamic_cells = -1


print()
print(
    "=" * 70
)

print(
    "BUILDING ADAPTIVE 2.5D MAPS"
)

print(
    "=" * 70
)


for index, file_path in enumerate(
    prediction_files
):

    print()
    print(
        f"[{index + 1}/"
        f"{len(prediction_files)}] "
        f"{os.path.basename(file_path)}"
    )


    # --------------------------------------------------------
    # LOAD SAVED POINTNET RESULT
    # --------------------------------------------------------

    data = np.load(
        file_path
    )


    points = data[
        "points"
    ].astype(
        np.float32
    )


    predictions = data[
        "predictions"
    ].astype(
        np.int64
    )


    frame_id = int(
        data[
            "frame"
        ]
    )


    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if (
        len(points)
        !=
        len(predictions)
    ):

        print(
            "WARNING: point/prediction "
            "count mismatch. Skipping."
        )

        continue


    valid = np.isfinite(
        points
    ).all(
        axis=1
    )


    points = points[
        valid
    ]


    predictions = predictions[
        valid
    ]


    # --------------------------------------------------------
    # BUILD ADAPTIVE GRID
    # --------------------------------------------------------

    cells = create_adaptive_cells(
        points,
        predictions
    )


    dynamic_cells = sum(

        1

        for cell in cells

        if cell["class"] == 3
    )


    # --------------------------------------------------------
    # SAVE FRAME
    # --------------------------------------------------------

    image_path = os.path.join(

        FRAME_DIR,

        f"semantic_adaptive_"
        f"{index:06d}.png"
    )


    draw_map(
        cells,
        frame_id,
        image_path
    )


    # --------------------------------------------------------
    # CLASS COUNTS
    # --------------------------------------------------------

    class_counts = np.bincount(
        predictions,
        minlength=7
    )


    summary_rows.append(
        [
            index,
            frame_id,
            len(points),
            len(cells),
            dynamic_cells,

            class_counts[0],
            class_counts[1],
            class_counts[2],
            class_counts[3],
            class_counts[4],
            class_counts[5],
            class_counts[6]
        ]
    )


    print(
        "Points:",
        len(points)
    )


    print(
        "Adaptive cells:",
        len(cells)
    )


    print(
        "Dynamic cells:",
        dynamic_cells
    )


    # --------------------------------------------------------
    # BEST FRAME
    # --------------------------------------------------------

    if dynamic_cells > (
        best_dynamic_cells
    ):

        best_dynamic_cells = (
            dynamic_cells
        )


        best_frame = {
            "index":
                index,

            "frame":
                frame_id,

            "cells":
                cells
        }


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_path = os.path.join(

    OUTPUT_DIR,

    "semantic_adaptive_summary.csv"
)


with open(
    summary_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow(
        [
            "index",
            "carla_frame",
            "points",
            "adaptive_cells",
            "dynamic_cells",

            "drivable",
            "non_drivable",
            "static_obstacle",
            "dynamic_object",
            "vegetation",
            "road_marking",
            "other"
        ]
    )


    writer.writerows(
        summary_rows
    )


# ============================================================
# BEST FRAME
# ============================================================

if best_frame is not None:

    best_path = os.path.join(

        OUTPUT_DIR,

        "best_semantic_adaptive_2_5d.png"
    )


    draw_map(

        best_frame[
            "cells"
        ],

        best_frame[
            "frame"
        ],

        best_path,

        show_height=True
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("SEMANTIC ADAPTIVE 2.5D COMPLETE")
print("=" * 70)

print(
    "Frames processed:",
    len(summary_rows)
)

print(
    "Summary:",
    os.path.abspath(
        summary_path
    )
)


if best_frame is not None:

    print(
        "Best map:",
        os.path.abspath(
            best_path
        )
    )

    print(
        "Best dynamic cells:",
        best_dynamic_cells
    )


print()
print(
    "Generated frame maps:",
    os.path.abspath(
        FRAME_DIR
    )
)

print()
print(
    "No PointNet++ inference was performed "
    "during this mapping stage."
)