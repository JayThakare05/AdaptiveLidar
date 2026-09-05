import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. LOAD EXISTING LiDAR MAP
# ============================================================

filename = "carla_lidar_map.ply"

print("Loading:", os.path.abspath(filename))

if not os.path.exists(filename):
    print("ERROR: carla_lidar_map.ply not found.")
    exit()


pcd = o3d.io.read_point_cloud(filename)

points = np.asarray(pcd.points)

print()
print("============================================")
print("3D LiDAR MAP LOADED")
print("============================================")
print("Total points:", len(points))


if len(points) == 0:
    print("ERROR: Point cloud is empty.")
    exit()


# ============================================================
# 2. EXTRACT X, Y, Z
# ============================================================

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


print()
print("X range:", x.min(), "to", x.max())
print("Y range:", y.min(), "to", y.max())
print("Z range:", z.min(), "to", z.max())


# ============================================================
# 3. DEFINE 2.5D GRID RESOLUTION
# ============================================================

# Each grid cell represents:
#
# GRID_SIZE × GRID_SIZE meters

GRID_SIZE = 0.5

print()
print("2.5D grid resolution:", GRID_SIZE, "meters")


# ============================================================
# 4. CALCULATE GRID BOUNDARIES
# ============================================================

min_x = x.min()
max_x = x.max()

min_y = y.min()
max_y = y.max()


grid_width = int(
    np.ceil(
        (max_x - min_x) / GRID_SIZE
    )
) + 1


grid_height = int(
    np.ceil(
        (max_y - min_y) / GRID_SIZE
    )
) + 1


print("Grid width:", grid_width)
print("Grid height:", grid_height)


# ============================================================
# 5. CREATE 2.5D ELEVATION GRID
# ============================================================

# Start with NaN because some cells
# may contain no LiDAR points.

height_map = np.full(
    (grid_height, grid_width),
    np.nan
)


# ============================================================
# 6. CONVERT POINTS → GRID CELLS
# ============================================================

grid_x = (
    (x - min_x) / GRID_SIZE
).astype(int)


grid_y = (
    (y - min_y) / GRID_SIZE
).astype(int)


# ============================================================
# 7. STORE HEIGHT FOR EACH CELL
# ============================================================

# If multiple LiDAR points fall into
# the same cell, we use the maximum Z.
#
# This gives us the highest detected
# surface/object in that cell.

for gx, gy, point_z in zip(
    grid_x,
    grid_y,
    z
):

    if np.isnan(
        height_map[gy, gx]
    ):

        height_map[gy, gx] = point_z

    else:

        height_map[gy, gx] = max(
            height_map[gy, gx],
            point_z
        )


# ============================================================
# 8. COUNT OCCUPIED CELLS
# ============================================================

occupied_cells = np.count_nonzero(
    ~np.isnan(height_map)
)

total_cells = height_map.size

print()
print("============================================")
print("2.5D MAP GENERATED")
print("============================================")

print("Total grid cells:", total_cells)

print(
    "Occupied cells:",
    occupied_cells
)

print(
    "Coverage:",
    round(
        occupied_cells / total_cells * 100,
        2
    ),
    "%"
)


# ============================================================
# 9. SAVE HEIGHT MAP
# ============================================================

np.save(
    "carla_2_5d_height_map.npy",
    height_map
)

print()
print(
    "Height map saved as:"
)

print(
    "carla_2_5d_height_map.npy"
)


# ============================================================
# 10. SAVE GRID AS CSV
# ============================================================

np.savetxt(
    "carla_2_5d_height_map.csv",
    height_map,
    delimiter=",",
    fmt="%.3f"
)

print(
    "CSV saved as:"
)

print(
    "carla_2_5d_height_map.csv"
)


# ============================================================
# 11. CREATE TOP-DOWN 2.5D MAP
# ============================================================

plt.figure(
    figsize=(12, 8)
)

plt.imshow(
    height_map,
    origin="lower",
    cmap="terrain",
    interpolation="nearest"
)

plt.colorbar(
    label="Elevation / Height (m)"
)

plt.title(
    "CARLA LiDAR - 2.5D Elevation Map"
)

plt.xlabel(
    "X Grid Cell"
)

plt.ylabel(
    "Y Grid Cell"
)

plt.tight_layout()


# ============================================================
# 12. SAVE 2.5D IMAGE
# ============================================================

plt.savefig(
    "carla_2_5d_map.png",
    dpi=200
)

print()
print(
    "2.5D map image saved as:"
)

print(
    "carla_2_5d_map.png"
)


# ============================================================
# 13. SHOW MAP
# ============================================================

plt.show()


print()
print("============================================")
print("2.5D MAPPING COMPLETE")
print("============================================")