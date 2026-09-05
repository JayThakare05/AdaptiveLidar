import carla
import time
import random
import numpy as np
import open3d as o3d


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_TIME = 5.0

LIDAR_RANGE = 50.0

VOXEL_SIZE = 0.10


# ============================================================
# CONNECT TO CARLA
# ============================================================

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

print("Connected to CARLA")
print("Map:", world.get_map().name)


# ============================================================
# BLUEPRINTS
# ============================================================

blueprints = world.get_blueprint_library()

vehicle_bp = blueprints.find(
    "vehicle.tesla.model3"
)

spawn_points = world.get_map().get_spawn_points()

random.shuffle(spawn_points)


# ============================================================
# SPAWN TESLA
# ============================================================

vehicle = None

for spawn_point in spawn_points:

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_point
    )

    if vehicle is not None:
        break


if vehicle is None:

    print("ERROR: Could not spawn Tesla.")

    exit()


print()
print("Tesla spawned:", vehicle.id)

# Stationary vehicle
vehicle.set_autopilot(False)


# ============================================================
# POSITION CAMERA BEHIND TESLA
# ============================================================

vehicle_transform = vehicle.get_transform()

# CARLA local coordinates:
# x = forward
# y = sideways
# z = up
#
# x = -10 means behind the vehicle
# z = 4 means camera is above the vehicle

camera_location = vehicle_transform.transform(
    carla.Location(
        x=-10.0,
        y=0.0,
        z=4.0
    )
)

camera_rotation = carla.Rotation(
    pitch=-15.0,
    yaw=vehicle_transform.rotation.yaw + 180.0,
    roll=0.0
)

spectator = world.get_spectator()

spectator.set_transform(
    carla.Transform(
        camera_location,
        camera_rotation
    )
)

print("Spectator positioned behind Tesla.")


# ============================================================
# ADD VISIBLE LiDAR MARKER
# ============================================================

lidar_marker_location = vehicle_transform.transform(
    carla.Location(
        x=0.0,
        y=0.0,
        z=2.5
    )
)

world.debug.draw_point(
    lidar_marker_location,
    size=0.35,
    color=carla.Color(
        r=255,
        g=0,
        b=0
    ),
    life_time=COLLECTION_TIME + 10
)

print("LiDAR marker placed.")


# ============================================================
# CREATE LiDAR
# ============================================================

lidar_bp = blueprints.find(
    "sensor.lidar.ray_cast"
)

lidar_bp.set_attribute(
    "channels",
    "32"
)

lidar_bp.set_attribute(
    "range",
    str(LIDAR_RANGE)
)

lidar_bp.set_attribute(
    "points_per_second",
    "56000"
)

lidar_bp.set_attribute(
    "rotation_frequency",
    "10"
)


lidar_transform = carla.Transform(
    carla.Location(
        x=0.0,
        y=0.0,
        z=2.5
    )
)


lidar = world.spawn_actor(
    lidar_bp,
    lidar_transform,
    attach_to=vehicle
)


print(
    "LiDAR attached:",
    lidar.id
)


# ============================================================
# COLLECT POINT CLOUD
# ============================================================

all_points = []

start_time = time.time()


def lidar_callback(data):

    # Read raw LiDAR data
    points = np.frombuffer(
        data.raw_data,
        dtype=np.float32
    )

    points = points.reshape((-1, 4))

    # XYZ only
    xyz = points[:, :3]

    # Convert to float64 before any calculations.
    # This prevents overflow during distance calculations.
    xyz = xyz.astype(np.float64)

    # --------------------------------------------------------
    # REMOVE NaN / INFINITY
    # --------------------------------------------------------

    valid = np.isfinite(xyz).all(axis=1)

    xyz = xyz[valid]

    # --------------------------------------------------------
    # REMOVE IMPOSSIBLE LiDAR VALUES
    # --------------------------------------------------------

    # LiDAR range is 50 m, so no point should be farther
    # than 50 m from the sensor.

    distance = np.sqrt(
        np.sum(
            xyz * xyz,
            axis=1
        )
    )

    valid = (
        distance <= LIDAR_RANGE
    )

    xyz = xyz[valid]

    # --------------------------------------------------------
    # REMOVE POINTS VERY CLOSE TO SENSOR
    # --------------------------------------------------------

    # These are likely parts of the Tesla itself.

    horizontal_distance = np.sqrt(
        xyz[:, 0] ** 2 +
        xyz[:, 1] ** 2
    )

    valid = (
        horizontal_distance > 2.0
    )

    xyz = xyz[valid]

    if len(xyz) > 0:

        all_points.append(xyz)

    print(
        f"Frame {data.frame}: "
        f"{len(xyz)} valid points"
    )


# ============================================================
# START LiDAR
# ============================================================

lidar.listen(
    lidar_callback
)


print()
print("============================================")
print("STARTING STATIONARY LiDAR COLLECTION")
print("============================================")
print()
print("Tesla is stationary.")
print("Look at the CARLA window to see the Tesla.")
print()
print(
    "Collecting for",
    COLLECTION_TIME,
    "seconds..."
)
print()


# ============================================================
# WAIT
# ============================================================

while (
    time.time() - start_time
    < COLLECTION_TIME
):

    time.sleep(0.1)


# ============================================================
# STOP SENSOR
# ============================================================

lidar.stop()

time.sleep(0.5)


# ============================================================
# CHECK DATA
# ============================================================

if len(all_points) == 0:

    print()
    print("ERROR: No valid LiDAR data collected.")

    lidar.destroy()
    vehicle.destroy()

    exit()


# ============================================================
# COMBINE POINTS
# ============================================================

points = np.vstack(
    all_points
)


print()
print("============================================")
print("LiDAR COLLECTION COMPLETE")
print("============================================")

print(
    "Frames collected:",
    len(all_points)
)

print(
    "Total valid points:",
    len(points)
)


# ============================================================
# FINAL SANITY CHECK
# ============================================================

valid = np.isfinite(points).all(axis=1)

points = points[valid]


distance = np.sqrt(
    np.sum(
        points * points,
        axis=1
    )
)

valid = (
    distance <= LIDAR_RANGE
)

points = points[valid]


print(
    "Points after final validation:",
    len(points)
)


# ============================================================
# OPEN3D POINT CLOUD
# ============================================================

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    points
)


# ============================================================
# VOXEL DOWNSAMPLING
# ============================================================

pcd = pcd.voxel_down_sample(
    voxel_size=VOXEL_SIZE
)


print(
    "Points after voxel downsampling:",
    len(pcd.points)
)


# ============================================================
# SAVE
# ============================================================

filename = (
    "stationary_lidar_map.ply"
)

success = o3d.io.write_point_cloud(
    filename,
    pcd
)


if success:

    print()
    print(
        "Saved:",
        filename
    )

else:

    print()
    print(
        "WARNING: Failed to save PLY."
    )


# ============================================================
# VISUALIZE
# ============================================================

print()
print("Opening Open3D viewer...")

o3d.visualization.draw_geometries(
    [pcd],
    window_name=(
        "Stationary CARLA LiDAR Map"
    )
)


# ============================================================
# CLEANUP
# ============================================================

lidar.destroy()

vehicle.destroy()

print()
print("Cleanup complete.")