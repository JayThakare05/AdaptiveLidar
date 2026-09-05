import carla
import time
import random
import numpy as np
import open3d as o3d
import os


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_TIME = 20.0

LIDAR_RANGE = 50.0

VOXEL_SIZE = 0.10

CAMERA_DISTANCE = 10.0
CAMERA_HEIGHT = 4.0

PRINT_EVERY_N_FRAMES = 20


# ============================================================
# CONNECT TO CARLA
# ============================================================

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

print("Connected to CARLA")
print("Map:", world.get_map().name)


# ============================================================
# KEEP CARLA ASYNCHRONOUS
# ============================================================

original_settings = world.get_settings()

if original_settings.synchronous_mode:

    print("CARLA is in synchronous mode.")
    print("Switching to asynchronous mode...")

    new_settings = world.get_settings()
    new_settings.synchronous_mode = False
    world.apply_settings(new_settings)

else:

    print("CARLA is already asynchronous.")


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


# ============================================================
# TRAFFIC MANAGER
# ============================================================

traffic_manager = client.get_trafficmanager(
    8000
)

# Keep a reasonable distance from the vehicle ahead
traffic_manager.set_global_distance_to_leading_vehicle(
    5.0
)

# Slightly reduce speed for safer mapping
traffic_manager.global_percentage_speed_difference(
    20.0
)


# ============================================================
# LiDAR
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


lidar_relative_transform = carla.Transform(
    carla.Location(
        x=0.0,
        y=0.0,
        z=2.5
    )
)


lidar = world.spawn_actor(
    lidar_bp,
    lidar_relative_transform,
    attach_to=vehicle
)


print(
    "LiDAR attached:",
    lidar.id
)


# ============================================================
# VEHICLE BOUNDING BOX
# ============================================================

bbox = vehicle.bounding_box

print()
print("Tesla bounding box:")
print(
    "Length:",
    bbox.extent.x * 2,
    "m"
)

print(
    "Width:",
    bbox.extent.y * 2,
    "m"
)

print(
    "Height:",
    bbox.extent.z * 2,
    "m"
)


# ============================================================
# SENSOR -> VEHICLE TRANSFORM
# ============================================================

vehicle_world_matrix = np.array(
    vehicle.get_transform().get_matrix()
)

lidar_world_matrix = np.array(
    lidar.get_transform().get_matrix()
)

vehicle_inverse_matrix = np.linalg.inv(
    vehicle_world_matrix
)

sensor_to_vehicle_matrix = (
    vehicle_inverse_matrix @
    lidar_world_matrix
)


# ============================================================
# DATA STORAGE
# ============================================================

all_world_points = []

trajectory = []

frame_counter = 0


# ============================================================
# LiDAR CALLBACK
# ============================================================

def lidar_callback(data):

    global frame_counter

    frame_counter += 1

    # --------------------------------------------------------
    # RAW LiDAR DATA
    # --------------------------------------------------------

    raw = np.frombuffer(
        data.raw_data,
        dtype=np.float32
    )

    points = raw.reshape(
        (-1, 4)
    )

    xyz = points[:, :3]

    # Convert to float64 before transformation
    xyz = xyz.astype(np.float64)


    # --------------------------------------------------------
    # REMOVE NaN / INF
    # --------------------------------------------------------

    valid = np.isfinite(
        xyz
    ).all(axis=1)

    xyz = xyz[valid]


    # --------------------------------------------------------
    # LiDAR RANGE CHECK
    # --------------------------------------------------------

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
    # TRANSFORM LiDAR POINTS
    # INTO VEHICLE COORDINATES
    # --------------------------------------------------------

    ones = np.ones(
        (len(xyz), 1),
        dtype=np.float64
    )

    homogeneous_points = np.hstack(
        (xyz, ones)
    )

    vehicle_points = (
        sensor_to_vehicle_matrix @
        homogeneous_points.T
    ).T[:, :3]


    # --------------------------------------------------------
    # REMOVE EGO VEHICLE
    # --------------------------------------------------------

    margin_x = 0.30
    margin_y = 0.30
    margin_z = 0.30

    min_x = (
        bbox.location.x -
        bbox.extent.x -
        margin_x
    )

    max_x = (
        bbox.location.x +
        bbox.extent.x +
        margin_x
    )

    min_y = (
        bbox.location.y -
        bbox.extent.y -
        margin_y
    )

    max_y = (
        bbox.location.y +
        bbox.extent.y +
        margin_y
    )

    min_z = (
        bbox.location.z -
        bbox.extent.z -
        margin_z
    )

    max_z = (
        bbox.location.z +
        bbox.extent.z +
        margin_z
    )


    inside_vehicle = (
        (vehicle_points[:, 0] >= min_x) &
        (vehicle_points[:, 0] <= max_x) &
        (vehicle_points[:, 1] >= min_y) &
        (vehicle_points[:, 1] <= max_y) &
        (vehicle_points[:, 2] >= min_z) &
        (vehicle_points[:, 2] <= max_z)
    )

    xyz = xyz[
        ~inside_vehicle
    ]


    # --------------------------------------------------------
    # TRANSFORM CLEAN POINTS INTO WORLD COORDINATES
    # --------------------------------------------------------

    ones = np.ones(
        (len(xyz), 1),
        dtype=np.float64
    )

    homogeneous_points = np.hstack(
        (xyz, ones)
    )


    world_matrix = np.array(
        data.transform.get_matrix()
    )


    world_points = (
        world_matrix @
        homogeneous_points.T
    ).T[:, :3]


    # --------------------------------------------------------
    # SAVE FRAME
    # --------------------------------------------------------

    if len(world_points) > 0:

        all_world_points.append(
            world_points
        )


    # --------------------------------------------------------
    # SAVE VEHICLE TRAJECTORY
    # --------------------------------------------------------

    sensor_transform = data.transform

    location = sensor_transform.location

    trajectory.append(
        [
            location.x,
            location.y,
            location.z
        ]
    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if frame_counter % PRINT_EVERY_N_FRAMES == 0:

        print(
            f"Frame {data.frame}: "
            f"{len(world_points)} mapped points"
        )


# ============================================================
# START LiDAR
# ============================================================

lidar.listen(
    lidar_callback
)


# ============================================================
# ENABLE AUTOPILOT
# ============================================================

vehicle.set_autopilot(
    True,
    traffic_manager.get_port()
)


print()
print("============================================")
print("EXPERIMENT B STARTED")
print("============================================")
print()
print("Tesla is now driving.")
print("LiDAR is being transformed into world coordinates.")
print("Ego-vehicle points are being removed.")
print()
print(
    "Collection time:",
    COLLECTION_TIME,
    "seconds"
)
print()


# ============================================================
# FOLLOWING CAMERA
# ============================================================

start_time = time.time()


while (
    time.time() - start_time
    < COLLECTION_TIME
):

    # Get current Tesla transform
    vehicle_transform = vehicle.get_transform()


    # Camera position behind Tesla
    camera_location = vehicle_transform.transform(
        carla.Location(
            x=-CAMERA_DISTANCE,
            y=0.0,
            z=CAMERA_HEIGHT
        )
    )


    # Camera faces toward Tesla
    camera_rotation = carla.Rotation(
        pitch=-15.0,
        yaw=vehicle_transform.rotation.yaw,
        roll=0.0
    )


    spectator = world.get_spectator()

    spectator.set_transform(
        carla.Transform(
            camera_location,
            camera_rotation
        )
    )


    time.sleep(0.10)


# ============================================================
# STOP
# ============================================================

lidar.stop()

time.sleep(0.5)


# ============================================================
# CHECK DATA
# ============================================================

print()
print("============================================")
print("EXPERIMENT B COMPLETE")
print("============================================")

print(
    "LiDAR frames processed:",
    frame_counter
)

if len(all_world_points) == 0:

    print(
        "ERROR: No valid world points collected."
    )

    lidar.destroy()
    vehicle.destroy()

    exit()


# ============================================================
# COMBINE ALL POINTS
# ============================================================

world_points = np.vstack(
    all_world_points
)


print(
    "Total accumulated points:",
    len(world_points)
)


# ============================================================
# FINAL VALIDATION
# ============================================================

valid = np.isfinite(
    world_points
).all(axis=1)

world_points = world_points[valid]


print(
    "Points after validation:",
    len(world_points)
)


# ============================================================
# CREATE OPEN3D CLOUD
# ============================================================

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    world_points
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
# STATISTICAL OUTLIER REMOVAL
# ============================================================

if len(pcd.points) > 100:

    pcd, indices = (
        pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )
    )


print(
    "Points after outlier removal:",
    len(pcd.points)
)


# ============================================================
# SAVE MAP
# ============================================================

filename = (
    "moving_lidar_map.ply"
)

success = o3d.io.write_point_cloud(
    filename,
    pcd
)


if success:

    print()
    print(
        "Saved:",
        os.path.abspath(filename)
    )

else:

    print()
    print(
        "ERROR: Could not save PLY."
    )


# ============================================================
# SAVE TRAJECTORY
# ============================================================

if len(trajectory) > 0:

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64
    )

    np.save(
        "ego_trajectory.npy",
        trajectory
    )

    print(
        "Saved:",
        os.path.abspath(
            "ego_trajectory.npy"
        )
    )


# ============================================================
# OPEN3D VISUALIZATION
# ============================================================

print()
print(
    "Opening moving LiDAR map..."
)

o3d.visualization.draw_geometries(
    [pcd],
    window_name=(
        "CARLA Moving LiDAR World Map"
    )
)


# ============================================================
# CLEANUP
# ============================================================

lidar.destroy()

vehicle.destroy()

print()
print("Cleanup complete.")