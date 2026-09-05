import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PLY_FILE = "carla_lidar_map.ply"

GRID_SIZE = 0.5          # meters per grid cell

# Ignore extremely high/low values that are likely noise
MIN_Z = -2.0
MAX_Z = 15.0

# Minimum obstacle height above estimated ground
OBSTACLE_HEIGHT_THRESHOLD = 0.50


# ============================================================
# LOAD POINT CLOUD
# ============================================================

print("Loading:", os.path.abspath(PLY_FILE))

if not os.path.exists(PLY_FILE):
    print("ERROR: PLY file not found.")
    exit()

pcd = o3d.io.read_point_cloud(PLY_FILE)

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
# EXTRACT XYZ
# ============================================================

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


print()
print("Original ranges:")
print("X:", x.min(), "to", x.max())
print("Y:", y.min(), "to", y.max())
print("Z:", z.min(), "to", z.max())


# ============================================================
# BASIC HEIGHT FILTER
# ============================================================

valid = (
    (z >= MIN_Z) &
    (z <= MAX_Z) &
    np.isfinite(x) &
    np.isfinite(y) &
    np.isfinite(z)
)

x = x[valid]
y = y[valid]
z = z[valid]

print()
print("Points after basic filtering:", len(x))


# ============================================================
# CREATE GRID
# ============================================================

x_min = x.min()
x_max = x.max()

y_min = y.min()
y_max = y.max()

grid_width = int(np.ceil((x_max - x_min) / GRID_SIZE)) + 1
grid_height = int(np.ceil((y_max - y_min) / GRID_SIZE)) + 1

print()
print("============================================")
print("2.5D GRID")
print("============================================")

print("Grid resolution:", GRID_SIZE, "meters")
print("Grid width:", grid_width)
print("Grid height:", grid_height)


# ============================================================
# CONVERT POINTS TO GRID INDICES
# ============================================================

ix = ((x - x_min) / GRID_SIZE).astype(int)
iy = ((y - y_min) / GRID_SIZE).astype(int)


# ============================================================
# COLLECT HEIGHT VALUES PER CELL
# ============================================================

cell_values = {}

for i in range(len(z)):

    cell = (ix[i], iy[i])

    if cell not in cell_values:
        cell_values[cell] = []

    cell_values[cell].append(z[i])


# ============================================================
# CREATE MAPS
# ============================================================

ground_map = np.full(
    (grid_height, grid_width),
    np.nan,
    dtype=np.float32
)

max_height_map = np.full(
    (grid_height, grid_width),
    np.nan,
    dtype=np.float32
)

obstacle_height_map = np.full(
    (grid_height, grid_width),
    np.nan,
    dtype=np.float32
)

point_count_map = np.zeros(
    (grid_height, grid_width),
    dtype=np.int32
)


# ============================================================
# CALCULATE GROUND + MAX HEIGHT
# ============================================================

for (cell_x, cell_y), values in cell_values.items():

    values = np.asarray(values)

    # Lower percentile gives a more stable estimate
    # of the ground than simply taking minimum Z.
    ground_z = np.percentile(values, 20)

    max_z = np.max(values)

    ground_map[cell_y, cell_x] = ground_z

    max_height_map[cell_y, cell_x] = max_z

    obstacle_height_map[cell_y, cell_x] = max_z - ground_z

    point_count_map[cell_y, cell_x] = len(values)


# ============================================================
# OCCUPANCY MAP
# ============================================================

occupied_map = (
    obstacle_height_map >= OBSTACLE_HEIGHT_THRESHOLD
)


# ============================================================
# STATISTICS
# ============================================================

total_cells = grid_width * grid_height

occupied_cells = np.sum(~np.isnan(ground_map))

obstacle_cells = np.sum(occupied_map)

coverage = (
    occupied_cells / total_cells
) * 100

obstacle_percentage = (
    obstacle_cells / total_cells
) * 100


print()
print("============================================")
print("2.5D MAP GENERATED")
print("============================================")

print("Total grid cells:", total_cells)

print("Observed cells:", occupied_cells)

print(
    "Coverage:",
    round(coverage, 2),
    "%"
)

print(
    "Obstacle cells:",
    obstacle_cells
)

print(
    "Obstacle coverage:",
    round(obstacle_percentage, 2),
    "%"
)


# ============================================================
# SAVE NUMPY FILES
# ============================================================

np.save(
    "carla_ground_map.npy",
    ground_map
)

np.save(
    "carla_obstacle_height_map.npy",
    obstacle_height_map
)

np.save(
    "carla_occupancy_map.npy",
    occupied_map
)

np.save(
    "carla_point_count_map.npy",
    point_count_map
)


print()
print("Saved:")
print("carla_ground_map.npy")
print("carla_obstacle_height_map.npy")
print("carla_occupancy_map.npy")
print("carla_point_count_map.npy")


# ============================================================
# SAVE CSV
# ============================================================

np.savetxt(
    "carla_obstacle_height_map.csv",
    obstacle_height_map,
    delimiter=","
)

print("carla_obstacle_height_map.csv")


# ============================================================
# VISUALIZATION
# ============================================================

fig = plt.figure(
    figsize=(12, 8)
)

ax = fig.add_subplot(111)


image = ax.imshow(
    obstacle_height_map,
    origin="lower",
    interpolation="nearest",
    cmap="terrain"
)


ax.set_title(
    "CARLA LiDAR - 2.5D Obstacle Height Map"
)

ax.set_xlabel(
    "X Grid Cell"
)

ax.set_ylabel(
    "Y Grid Cell"
)


cbar = plt.colorbar(image, ax=ax)

cbar.set_label(
    "Obstacle Height Above Ground (m)"
)


plt.tight_layout()


plt.savefig(
    "carla_2_5d_obstacle_map.png",
    dpi=200
)


print()
print("Image saved as:")
print("carla_2_5d_obstacle_map.png")


plt.show()


# ============================================================
# OCCUPANCY VISUALIZATION
# ============================================================

plt.figure(
    figsize=(12, 8)
)

plt.imshow(
    occupied_map,
    origin="lower",
    interpolation="nearest"
)

plt.title(
    "CARLA LiDAR - 2.5D Occupancy Map"
)

plt.xlabel(
    "X Grid Cell"
)

plt.ylabel(
    "Y Grid Cell"
)

plt.tight_layout()

plt.savefig(
    "carla_2_5d_occupancy_map.png",
    dpi=200
)

print(
    "Image saved as:"
)

print(
    "carla_2_5d_occupancy_map.png"
)

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================")
print("IMPROVED 2.5D MAPPING COMPLETE")
print("============================================")