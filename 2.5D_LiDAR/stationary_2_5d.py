import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PLY_FILE = "stationary_lidar_map.ply"

GRID_SIZE = 0.5

# Ignore points below/above realistic scene limits
MIN_Z = -2.0
MAX_Z = 15.0

# Height above estimated ground required to call something
# an obstacle
OBSTACLE_THRESHOLD = 0.50


# ============================================================
# LOAD PLY
# ============================================================

print("Loading:", os.path.abspath(PLY_FILE))

if not os.path.exists(PLY_FILE):
    print("ERROR: stationary_lidar_map.ply not found.")
    exit()

pcd = o3d.io.read_point_cloud(PLY_FILE)

points = np.asarray(pcd.points)

print()
print("============================================")
print("STATIONARY LiDAR MAP")
print("============================================")

print("Points:", len(points))

if len(points) == 0:
    print("ERROR: Point cloud is empty.")
    exit()


# ============================================================
# COORDINATE CONVERSION
# ============================================================
#
# CARLA:
#   X = forward
#   Y = right
#   Z = up
#
# For our top-down map we explicitly flip Y so that
# the visual orientation is consistent with a conventional
# top-down right-handed representation.
#
# This makes the transformation intentional instead of
# relying on the Open3D viewer camera.
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
    (z >= MIN_Z) &
    (z <= MAX_Z)
)

x = x[valid]
y = y[valid]
z = z[valid]

print()
print("Points after filtering:", len(x))

print()
print("X range:", x.min(), "to", x.max())
print("Y range:", y.min(), "to", y.max())
print("Z range:", z.min(), "to", z.max())


# ============================================================
# GRID
# ============================================================

x_min = x.min()
x_max = x.max()

y_min = y.min()
y_max = y.max()

width = int(
    np.ceil((x_max - x_min) / GRID_SIZE)
) + 1

height = int(
    np.ceil((y_max - y_min) / GRID_SIZE)
) + 1


print()
print("============================================")
print("2.5D GRID")
print("============================================")

print("Resolution:", GRID_SIZE, "m")
print("Width:", width)
print("Height:", height)


# ============================================================
# GRID INDICES
# ============================================================

ix = (
    (x - x_min) / GRID_SIZE
).astype(int)

iy = (
    (y - y_min) / GRID_SIZE
).astype(int)


# ============================================================
# CELL STORAGE
# ============================================================

cells = {}

for i in range(len(z)):

    key = (ix[i], iy[i])

    if key not in cells:
        cells[key] = []

    cells[key].append(z[i])


# ============================================================
# MAP ARRAYS
# ============================================================

ground_map = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)

max_map = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)

obstacle_map = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)


# ============================================================
# BUILD MAP
# ============================================================

for (gx, gy), values in cells.items():

    values = np.asarray(values)

    # Lower percentile approximates the local ground
    ground = np.percentile(
        values,
        20
    )

    highest = np.max(values)

    obstacle_height = (
        highest - ground
    )

    ground_map[gy, gx] = ground

    max_map[gy, gx] = highest

    obstacle_map[gy, gx] = obstacle_height


# ============================================================
# OBSTACLE MAP
# ============================================================

occupancy_map = (
    obstacle_map >= OBSTACLE_THRESHOLD
)


# ============================================================
# STATISTICS
# ============================================================

observed = np.sum(
    ~np.isnan(ground_map)
)

obstacles = np.sum(
    occupancy_map
)

total = width * height

coverage = (
    observed / total
) * 100

obstacle_percentage = (
    obstacles / total
) * 100


print()
print("============================================")
print("RESULT")
print("============================================")

print(
    "Observed cells:",
    observed
)

print(
    "Coverage:",
    round(coverage, 2),
    "%"
)

print(
    "Obstacle cells:",
    obstacles
)

print(
    "Obstacle coverage:",
    round(obstacle_percentage, 2),
    "%"
)


# ============================================================
# SAVE DATA
# ============================================================

np.save(
    "stationary_ground_map.npy",
    ground_map
)

np.save(
    "stationary_obstacle_height_map.npy",
    obstacle_map
)

np.save(
    "stationary_occupancy_map.npy",
    occupancy_map
)


# ============================================================
# HEIGHT MAP
# ============================================================

plt.figure(
    figsize=(11, 8)
)

plt.imshow(
    obstacle_map,
    origin="lower",
    interpolation="nearest",
    cmap="terrain"
)

plt.title(
    "Stationary CARLA LiDAR - 2.5D Obstacle Height Map"
)

plt.xlabel(
    "X Grid Cell"
)

plt.ylabel(
    "Y Grid Cell"
)

plt.colorbar(
    label="Obstacle Height Above Ground (m)"
)

plt.tight_layout()

plt.savefig(
    "stationary_2_5d_height_map.png",
    dpi=200
)

plt.show()


# ============================================================
# OCCUPANCY MAP
# ============================================================

plt.figure(
    figsize=(11, 8)
)

plt.imshow(
    occupancy_map,
    origin="lower",
    interpolation="nearest"
)

plt.title(
    "Stationary CARLA LiDAR - 2.5D Occupancy Map"
)

plt.xlabel(
    "X Grid Cell"
)

plt.ylabel(
    "Y Grid Cell"
)

plt.tight_layout()

plt.savefig(
    "stationary_2_5d_occupancy_map.png",
    dpi=200
)

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================")
print("STATIONARY 2.5D MAPPING COMPLETE")
print("============================================")