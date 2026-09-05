import os
import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

PLY_FILE = "moving_lidar_map.ply"

# Ground estimation grid
GROUND_GRID_SIZE = 0.5

# Minimum height above local ground to become a candidate
GROUND_HEIGHT_THRESHOLD = 0.50

# Ignore tiny clusters
MIN_CLUSTER_POINTS = 30

# DBSCAN parameters
DBSCAN_EPS = 0.75

DBSCAN_MIN_POINTS = 12

# Ignore extremely large clusters.
# These are usually buildings / walls / accumulated structures.
MAX_OBJECT_LENGTH = 12.0
MAX_OBJECT_WIDTH = 8.0
MAX_OBJECT_HEIGHT = 6.0


# ============================================================
# LOAD POINT CLOUD
# ============================================================

print("Loading:", os.path.abspath(PLY_FILE))

if not os.path.exists(PLY_FILE):

    print("ERROR: moving_lidar_map.ply not found.")
    exit()


pcd = o3d.io.read_point_cloud(
    PLY_FILE
)

points = np.asarray(
    pcd.points
).astype(np.float64)


print()
print("============================================")
print("LiDAR OBJECT DETECTION")
print("============================================")

print(
    "Total points:",
    len(points)
)


if len(points) == 0:

    print("ERROR: Point cloud is empty.")
    exit()


# ============================================================
# BASIC VALIDATION
# ============================================================

valid = np.isfinite(
    points
).all(axis=1)

points = points[valid]


print(
    "Points after validation:",
    len(points)
)


# ============================================================
# EXTRACT XYZ
# ============================================================

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


# ============================================================
# BASIC Z FILTER
# ============================================================

valid = (
    (z >= -3.0) &
    (z <= 15.0)
)

points = points[valid]

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


print(
    "Points after Z filtering:",
    len(points)
)


# ============================================================
# LOCAL GROUND ESTIMATION
# ============================================================

print()
print("Estimating local ground...")


x_min = x.min()
y_min = y.min()


grid_x = (
    (x - x_min) /
    GROUND_GRID_SIZE
).astype(np.int32)


grid_y = (
    (y - y.min()) /
    GROUND_GRID_SIZE
).astype(np.int32)


# Dictionary:
# (grid_x, grid_y) -> list of Z values

cell_z = {}


for i in range(len(points)):

    key = (
        grid_x[i],
        grid_y[i]
    )

    if key not in cell_z:

        cell_z[key] = []

    cell_z[key].append(
        z[i]
    )


# ============================================================
# GROUND HEIGHT PER CELL
# ============================================================

ground_lookup = {}


for key, values in cell_z.items():

    values = np.asarray(
        values
    )

    # Low percentile provides an approximate
    # local ground level.
    ground = np.percentile(
        values,
        15
    )

    ground_lookup[key] = ground


# ============================================================
# REMOVE GROUND
# ============================================================

print(
    "Removing ground points..."
)


non_ground_indices = []


for i in range(len(points)):

    key = (
        grid_x[i],
        grid_y[i]
    )

    ground = ground_lookup.get(
        key
    )

    if ground is None:
        continue


    height_above_ground = (
        z[i] - ground
    )


    if (
        height_above_ground
        >= GROUND_HEIGHT_THRESHOLD
    ):

        non_ground_indices.append(
            i
        )


non_ground_points = points[
    non_ground_indices
]


print(
    "Non-ground points:",
    len(non_ground_points)
)


if len(non_ground_points) < MIN_CLUSTER_POINTS:

    print(
        "ERROR: Not enough non-ground points."
    )

    exit()


# ============================================================
# CREATE OPEN3D CLOUD
# ============================================================

object_cloud = (
    o3d.geometry.PointCloud()
)

object_cloud.points = (
    o3d.utility.Vector3dVector(
        non_ground_points
    )
)


# ============================================================
# DBSCAN CLUSTERING
# ============================================================

print()
print("Running DBSCAN clustering...")


labels = np.array(
    object_cloud.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=True
    )
)


# Number of clusters
num_clusters = (
    labels.max() + 1
)


print()
print(
    "Clusters found:",
    num_clusters
)


# ============================================================
# EXTRACT OBJECTS
# ============================================================

objects = []


for cluster_id in range(
    num_clusters
):

    cluster_indices = np.where(
        labels == cluster_id
    )[0]


    if len(cluster_indices) < (
        MIN_CLUSTER_POINTS
    ):

        continue


    cluster_points = (
        non_ground_points[
            cluster_indices
        ]
    )


    # --------------------------------------------------------
    # BOUNDING BOX
    # --------------------------------------------------------

    min_bound = (
        cluster_points.min(
            axis=0
        )
    )

    max_bound = (
        cluster_points.max(
            axis=0
        )
    )


    dimensions = (
        max_bound -
        min_bound
    )


    length = dimensions[0]
    width = dimensions[1]
    height = dimensions[2]


    # --------------------------------------------------------
    # REMOVE GIANT STRUCTURES
    # --------------------------------------------------------

    if (
        length > MAX_OBJECT_LENGTH
        or
        width > MAX_OBJECT_WIDTH
        or
        height > MAX_OBJECT_HEIGHT
    ):

        continue


    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    center = (
        min_bound +
        max_bound
    ) / 2.0


    objects.append(
        {
            "id": len(objects) + 1,
            "points": len(cluster_points),
            "center": center,
            "dimensions": dimensions,
            "min": min_bound,
            "max": max_bound
        }
    )


# ============================================================
# PRINT DETECTIONS
# ============================================================

print()
print("============================================")
print("DETECTED OBJECTS")
print("============================================")


if len(objects) == 0:

    print(
        "No usable objects detected."
    )

else:

    for obj in objects:

        center = obj["center"]

        dimensions = obj["dimensions"]


        print()
        print(
            f"Object {obj['id']}"
        )

        print(
            "Points:",
            obj["points"]
        )

        print(
            "Center:",
            "(",
            round(center[0], 2),
            ",",
            round(center[1], 2),
            ",",
            round(center[2], 2),
            ")"
        )

        print(
            "Dimensions:",
            "(",
            round(dimensions[0], 2),
            "m x",
            round(dimensions[1], 2),
            "m x",
            round(dimensions[2], 2),
            "m"
            ")"
        )


print()
print(
    "Usable detected objects:",
    len(objects)
)


# ============================================================
# CREATE OBJECT BOUNDING BOXES
# ============================================================

geometries = []


# Original non-ground cloud
geometries.append(
    object_cloud
)


for obj in objects:

    min_bound = obj["min"]
    max_bound = obj["max"]


    bbox = (
        o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound,
            max_bound=max_bound
        )
    )


    # Make bounding boxes visually obvious
    bbox.color = (
        1.0,
        0.0,
        0.0
    )


    geometries.append(
        bbox
    )


# ============================================================
# VISUALIZE
# ============================================================

print()
print(
    "Opening object detection viewer..."
)


o3d.visualization.draw_geometries(
    geometries,
    window_name=(
        "CARLA LiDAR - Detected Objects"
    )
)


# ============================================================
# SAVE DETECTIONS
# ============================================================

with open(
    "detected_objects.csv",
    "w"
) as file:

    file.write(
        "object_id,"
        "points,"
        "center_x,"
        "center_y,"
        "center_z,"
        "length,"
        "width,"
        "height\n"
    )


    for obj in objects:

        center = obj["center"]

        dimensions = obj["dimensions"]


        file.write(
            f"{obj['id']},"
            f"{obj['points']},"
            f"{center[0]:.3f},"
            f"{center[1]:.3f},"
            f"{center[2]:.3f},"
            f"{dimensions[0]:.3f},"
            f"{dimensions[1]:.3f},"
            f"{dimensions[2]:.3f}\n"
        )


print()
print(
    "Saved: detected_objects.csv"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================")
print("OBJECT EXTRACTION COMPLETE")
print("============================================")