import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIG
# ============================================================

PLY_FILE = "moving_lidar_map.ply"

BASE_CELL = 2.0

LEVEL1_CELL = 1.0

LEVEL2_CELL = 0.5

OBSTACLE_THRESHOLD = 0.75

HEIGHT_VARIATION_THRESHOLD = 0.60

MIN_POINTS = 5


# ============================================================
# LOAD POINT CLOUD
# ============================================================

print("Loading:", os.path.abspath(PLY_FILE))

if not os.path.exists(PLY_FILE):
    print("ERROR: moving_lidar_map.ply not found.")
    exit()

pcd = o3d.io.read_point_cloud(PLY_FILE)

points = np.asarray(pcd.points)

print()
print("============================================")
print("MOVING LiDAR MAP")
print("============================================")

print("Total points:", len(points))

if len(points) == 0:
    print("ERROR: Empty point cloud.")
    exit()


# ============================================================
# XYZ
# ============================================================

x = points[:, 0]
y = -points[:, 1]
z = points[:, 2]


# ============================================================
# FILTER
# ============================================================

valid = (
    np.isfinite(x) &
    np.isfinite(y) &
    np.isfinite(z) &
    (z > -3.0) &
    (z < 15.0)
)

x = x[valid]
y = y[valid]
z = z[valid]

print("Points after filtering:", len(x))


# ============================================================
# GLOBAL BOUNDS
# ============================================================

x_min = x.min()
x_max = x.max()

y_min = y.min()
y_max = y.max()


print()
print("X:", x_min, "to", x_max)
print("Y:", y_min, "to", y_max)


# ============================================================
# BASE GRID
# ============================================================

base_width = int(
    np.ceil(
        (x_max - x_min) / BASE_CELL
    )
)

base_height = int(
    np.ceil(
        (y_max - y_min) / BASE_CELL
    )
)


print()
print("Base grid:")
print(
    base_width,
    "x",
    base_height
)


# ============================================================
# ADAPTIVE CELLS
# ============================================================

leaf_cells = []

obstacle_cells = []


for bx in range(base_width):

    for by in range(base_height):

        bx_min = (
            x_min +
            bx * BASE_CELL
        )

        bx_max = (
            bx_min +
            BASE_CELL
        )

        by_min = (
            y_min +
            by * BASE_CELL
        )

        by_max = (
            by_min +
            BASE_CELL
        )


        mask = (
            (x >= bx_min) &
            (x < bx_max) &
            (y >= by_min) &
            (y < by_max)
        )

        indices = np.where(mask)[0]


        if len(indices) < MIN_POINTS:
            continue


        cell_z = z[indices]

        ground = np.percentile(
            cell_z,
            20
        )

        highest = np.percentile(
            cell_z,
            95
        )

        obstacle_height = (
            highest - ground
        )

        variation = (
            np.percentile(cell_z, 90) -
            np.percentile(cell_z, 10)
        )


        # ----------------------------------------------------
        # DECIDE RESOLUTION
        # ----------------------------------------------------

        complex_region = (
            obstacle_height >
            OBSTACLE_THRESHOLD
            or
            variation >
            HEIGHT_VARIATION_THRESHOLD
        )


        # ----------------------------------------------------
        # SIMPLE REGION
        # ----------------------------------------------------

        if not complex_region:

            leaf_cells.append(
                (
                    bx_min,
                    by_min,
                    BASE_CELL,
                    ground,
                    obstacle_height,
                    len(indices)
                )
            )

            continue


        # ====================================================
        # LEVEL 1 — 1m
        # ====================================================

        sub_size = LEVEL1_CELL

        sub_count = int(
            BASE_CELL / sub_size
        )


        for sx in range(sub_count):

            for sy in range(sub_count):

                sx_min = (
                    bx_min +
                    sx * sub_size
                )

                sx_max = (
                    sx_min +
                    sub_size
                )

                sy_min = (
                    by_min +
                    sy * sub_size
                )

                sy_max = (
                    sy_min +
                    sub_size
                )


                submask = (
                    (x >= sx_min) &
                    (x < sx_max) &
                    (y >= sy_min) &
                    (y < sy_max)
                )

                sub_indices = np.where(
                    submask
                )[0]


                if len(sub_indices) < MIN_POINTS:
                    continue


                sub_z = z[sub_indices]

                sub_ground = np.percentile(
                    sub_z,
                    20
                )

                sub_high = np.percentile(
                    sub_z,
                    95
                )

                sub_obstacle = (
                    sub_high -
                    sub_ground
                )


                # --------------------------------------------
                # HIGH COMPLEXITY?
                # --------------------------------------------

                if sub_obstacle > 1.5:

                    # ========================================
                    # LEVEL 2 — 0.5m
                    # ========================================

                    fine_size = LEVEL2_CELL

                    fine_count = int(
                        sub_size / fine_size
                    )


                    for fx in range(fine_count):

                        for fy in range(fine_count):

                            fx_min = (
                                sx_min +
                                fx * fine_size
                            )

                            fx_max = (
                                fx_min +
                                fine_size
                            )

                            fy_min = (
                                sy_min +
                                fy * fine_size
                            )

                            fy_max = (
                                fy_min +
                                fine_size
                            )


                            fine_mask = (
                                (x >= fx_min) &
                                (x < fx_max) &
                                (y >= fy_min) &
                                (y < fy_max)
                            )

                            fine_indices = np.where(
                                fine_mask
                            )[0]


                            if len(fine_indices) < MIN_POINTS:
                                continue


                            fine_z = z[fine_indices]

                            fine_ground = np.percentile(
                                fine_z,
                                20
                            )

                            fine_high = np.percentile(
                                fine_z,
                                95
                            )

                            fine_obstacle = (
                                fine_high -
                                fine_ground
                            )


                            leaf_cells.append(
                                (
                                    fx_min,
                                    fy_min,
                                    fine_size,
                                    fine_ground,
                                    fine_obstacle,
                                    len(fine_indices)
                                )
                            )


                    continue


                # --------------------------------------------
                # NORMAL 1m CELL
                # --------------------------------------------

                leaf_cells.append(
                    (
                        sx_min,
                        sy_min,
                        sub_size,
                        sub_ground,
                        sub_obstacle,
                        len(sub_indices)
                    )
                )


# ============================================================
# STATISTICS
# ============================================================

print()
print("============================================")
print("ADAPTIVE 2.5D MAP")
print("============================================")

print(
    "Total adaptive cells:",
    len(leaf_cells)
)


base_count = 0
level1_count = 0
level2_count = 0


for cell in leaf_cells:

    size = cell[2]

    if size == BASE_CELL:
        base_count += 1

    elif size == LEVEL1_CELL:
        level1_count += 1

    elif size == LEVEL2_CELL:
        level2_count += 1


print(
    "2.0m cells:",
    base_count
)

print(
    "1.0m cells:",
    level1_count
)

print(
    "0.5m cells:",
    level2_count
)


# ============================================================
# SAVE CSV
# ============================================================

data = np.asarray(
    leaf_cells
)

np.savetxt(
    "adaptive_2_5d_cells.csv",
    data,
    delimiter=",",
    header=(
        "x,y,size,ground,"
        "obstacle_height,points"
    ),
    comments=""
)


print()
print(
    "Saved: adaptive_2_5d_cells.csv"
)


# ============================================================
# VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 10)
)


max_obstacle = max(
    cell[4]
    for cell in leaf_cells
)


for (
    cx,
    cy,
    size,
    ground,
    obstacle_height,
    count
) in leaf_cells:

    if obstacle_height >= OBSTACLE_THRESHOLD:

        normalized = min(
            obstacle_height /
            max_obstacle,
            1.0
        )

        facecolor = (
            normalized,
            0.2,
            1.0 - normalized
        )

        alpha = 0.85

    else:

        facecolor = "lightgray"

        alpha = 0.30


    rect = Rectangle(
        (
            cx,
            cy
        ),
        size,
        size,
        facecolor=facecolor,
        edgecolor="none",
        alpha=alpha
    )

    ax.add_patch(rect)


# ============================================================
# TRAJECTORY
# ============================================================

if os.path.exists(
    "ego_trajectory.npy"
):

    trajectory = np.load(
        "ego_trajectory.npy"
    )

    ax.plot(
        trajectory[:, 0],
        -trajectory[:, 1],
        linewidth=2,
        label="Tesla trajectory"
    )


# ============================================================
# AXES
# ============================================================

ax.set_xlim(
    x_min,
    x_max
)

ax.set_ylim(
    y_min,
    y_max
)

ax.set_aspect(
    "equal"
)

ax.set_title(
    "Adaptive 2.5D LiDAR Map"
)

ax.set_xlabel(
    "World X (m)"
)

ax.set_ylabel(
    "World Y (m)"
)

ax.legend()


plt.tight_layout()

plt.savefig(
    "adaptive_2_5d_map.png",
    dpi=200
)

plt.show()


print()
print(
    "Saved: adaptive_2_5d_map.png"
)

print()
print("============================================")
print("ADAPTIVE 2.5D COMPLETE")
print("============================================")