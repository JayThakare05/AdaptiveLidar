import carla
import time
import random
import os
import threading
import numpy as np
import open3d as o3d
import psutil


# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_TIME = 20.0

LIDAR_RANGE = 50.0

GRID_SIZE = 0.5

VOXEL_SIZE = 0.10

MEMORY_SAMPLE_INTERVAL = 0.01


# ============================================================
# PROCESS MEMORY MONITOR
# ============================================================

process = psutil.Process(
    os.getpid()
)

monitoring = False

peak_memory = 0.0


def monitor_memory():

    global peak_memory

    while monitoring:

        memory_mb = (
            process.memory_info().rss
            / (1024 * 1024)
        )

        if memory_mb > peak_memory:

            peak_memory = memory_mb

        time.sleep(
            MEMORY_SAMPLE_INTERVAL
        )


# ============================================================
# CONNECT TO CARLA
# ============================================================

client = carla.Client(
    "localhost",
    2000
)

client.set_timeout(10.0)

world = client.get_world()

print()
print("============================================")
print("3D vs 2.5D MAPPING BENCHMARK")
print("============================================")

print(
    "CARLA Map:",
    world.get_map().name
)


# ============================================================
# ENSURE ASYNC MODE
# ============================================================

settings = world.get_settings()

if settings.synchronous_mode:

    settings.synchronous_mode = False

    world.apply_settings(
        settings
    )

    print(
        "Switched CARLA to asynchronous mode."
    )

else:

    print(
        "CARLA already asynchronous."
    )


# ============================================================
# SPAWN EGO VEHICLE
# ============================================================

blueprints = (
    world.get_blueprint_library()
)

vehicle_bp = blueprints.find(
    "vehicle.tesla.model3"
)

spawn_points = (
    world.get_map().get_spawn_points()
)

random.shuffle(
    spawn_points
)

vehicle = None

for spawn_point in spawn_points:

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_point
    )

    if vehicle is not None:

        break


if vehicle is None:

    print(
        "ERROR: Could not spawn Tesla."
    )

    exit()


print(
    "Tesla spawned:",
    vehicle.id
)


# ============================================================
# TRAFFIC MANAGER
# ============================================================

traffic_manager = (
    client.get_trafficmanager(8000)
)

traffic_manager.set_global_distance_to_leading_vehicle(
    5.0
)

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


lidar = world.spawn_actor(
    lidar_bp,
    carla.Transform(
        carla.Location(
            x=0.0,
            y=0.0,
            z=2.5
        )
    ),
    attach_to=vehicle
)


print(
    "LiDAR attached:",
    lidar.id
)


# ============================================================
# COLLECT LiDAR DATA
# ============================================================

all_points = []

frame_count = 0


def lidar_callback(data):

    global frame_count

    frame_count += 1

    raw = np.frombuffer(
        data.raw_data,
        dtype=np.float32
    )

    raw = raw.reshape(
        (-1, 4)
    )

    xyz = raw[:, :3].astype(
        np.float64
    )


    # Remove invalid data

    valid = np.isfinite(
        xyz
    ).all(axis=1)

    xyz = xyz[valid]


    # Range filter

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


    if len(xyz) == 0:

        return


    # Transform LiDAR points
    # into world coordinates

    transform = np.array(
        data.transform.get_matrix()
    )

    ones = np.ones(
        (len(xyz), 1),
        dtype=np.float64
    )

    homogeneous = np.hstack(
        (xyz, ones)
    )

    world_points = (
        transform @
        homogeneous.T
    ).T[:, :3]


    all_points.append(
        world_points
    )


# ============================================================
# START COLLECTION
# ============================================================

lidar.listen(
    lidar_callback
)

vehicle.set_autopilot(
    True,
    traffic_manager.get_port()
)


print()
print(
    "--------------------------------------------"
)

print(
    "COLLECTING IDENTICAL DATASET"
)

print(
    "Drive time:",
    DRIVE_TIME,
    "seconds"
)

print(
    "--------------------------------------------"
)

print()


start_drive = time.perf_counter()

while (
    time.perf_counter() - start_drive
    < DRIVE_TIME
):

    # Follow vehicle with spectator

    vehicle_transform = (
        vehicle.get_transform()
    )

    camera_location = (
        vehicle_transform.transform(
            carla.Location(
                x=-14.0,
                y=0.0,
                z=6.0
            )
        )
    )

    camera_rotation = (
        carla.Rotation(
            pitch=-15.0,
            yaw=vehicle_transform.rotation.yaw,
            roll=0.0
        )
    )

    world.get_spectator().set_transform(
        carla.Transform(
            camera_location,
            camera_rotation
        )
    )

    time.sleep(0.05)


# ============================================================
# STOP COLLECTION
# ============================================================

lidar.stop()

time.sleep(0.5)


if len(all_points) == 0:

    print(
        "ERROR: No LiDAR points collected."
    )

    lidar.destroy()
    vehicle.destroy()

    exit()


points = np.vstack(
    all_points
)


print()
print(
    "============================================"
)

print(
    "DATASET COLLECTION COMPLETE"
)

print(
    "Frames:",
    frame_count
)

print(
    "Total points:",
    len(points)
)

print(
    "============================================"
)


# ============================================================
# SAVE COMMON INPUT
# ============================================================

o3d.io.write_point_cloud(
    "benchmark_input.ply",
    o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(
            points
        )
    )
)


# ============================================================
# FUNCTION: START MEMORY MONITOR
# ============================================================

def start_monitor():

    global monitoring
    global peak_memory

    peak_memory = (
        process.memory_info().rss
        / (1024 * 1024)
    )

    monitoring = True

    thread = threading.Thread(
        target=monitor_memory,
        daemon=True
    )

    thread.start()

    return thread


# ============================================================
# 3D MAPPING BENCHMARK
# ============================================================

print()
print(
    "============================================"
)

print(
    "3D MAPPING"
)

print(
    "============================================"
)


monitor_thread = start_monitor()

memory_before_3d = (
    process.memory_info().rss
    / (1024 * 1024)
)

start_3d = time.perf_counter()


# Create full 3D cloud

pcd_3d = o3d.geometry.PointCloud()

pcd_3d.points = (
    o3d.utility.Vector3dVector(
        points
    )
)


# Voxel processing

pcd_3d = (
    pcd_3d.voxel_down_sample(
        VOXEL_SIZE
    )
)


# Statistical filtering

if len(pcd_3d.points) > 100:

    pcd_3d, _ = (
        pcd_3d.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )
    )


# Save

o3d.io.write_point_cloud(
    "benchmark_3d_map.ply",
    pcd_3d
)


time_3d = (
    time.perf_counter() -
    start_3d
)


memory_after_3d = (
    process.memory_info().rss
    / (1024 * 1024)
)


monitoring = False

peak_3d = peak_memory


points_3d = len(
    pcd_3d.points
)


print(
    "3D points:",
    points_3d
)

print(
    "3D mapping time:",
    round(time_3d, 4),
    "seconds"
)

print(
    "3D memory before:",
    round(memory_before_3d, 2),
    "MB"
)

print(
    "3D memory after:",
    round(memory_after_3d, 2),
    "MB"
)

print(
    "3D peak memory:",
    round(peak_3d, 2),
    "MB"
)


# Free 3D map before 2.5D
del pcd_3d


# ============================================================
# 2.5D MAPPING BENCHMARK
# ============================================================

print()
print(
    "============================================"
)

print(
    "2.5D MAPPING"
)

print(
    "============================================"
)


monitor_thread = start_monitor()

memory_before_25d = (
    process.memory_info().rss
    / (1024 * 1024)
)

start_25d = time.perf_counter()


# ------------------------------------------------------------
# COORDINATES
# ------------------------------------------------------------

x = points[:, 0]

y = points[:, 1]

z = points[:, 2]


# ------------------------------------------------------------
# GRID BOUNDS
# ------------------------------------------------------------

x_min = x.min()
x_max = x.max()

y_min = y.min()
y_max = y.max()


width = (
    int(
        np.ceil(
            (x_max - x_min)
            / GRID_SIZE
        )
    )
    + 1
)


height = (
    int(
        np.ceil(
            (y_max - y_min)
            / GRID_SIZE
        )
    )
    + 1
)


# ------------------------------------------------------------
# GRID INDICES
# ------------------------------------------------------------

ix = (
    (x - x_min) /
    GRID_SIZE
).astype(np.int32)


iy = (
    (y - y_min) /
    GRID_SIZE
).astype(np.int32)


# ------------------------------------------------------------
# HEIGHT MAP
# ------------------------------------------------------------

height_map = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)


# For fair timing, calculate maximum
# height per cell.

for i in range(
    len(points)
):

    gx = ix[i]

    gy = iy[i]

    current = height_map[
        gy,
        gx
    ]

    value = z[i]


    if np.isnan(current):

        height_map[
            gy,
            gx
        ] = value

    elif value > current:

        height_map[
            gy,
            gx
        ] = value


# Save 2.5D map

np.save(
    "benchmark_2_5d_height_map.npy",
    height_map
)


time_25d = (
    time.perf_counter() -
    start_25d
)


memory_after_25d = (
    process.memory_info().rss
    / (1024 * 1024)
)


monitoring = False

peak_25d = peak_memory


occupied_cells = np.sum(
    ~np.isnan(height_map)
)


print(
    "2.5D grid:",
    width,
    "x",
    height
)

print(
    "Occupied cells:",
    occupied_cells
)

print(
    "2.5D mapping time:",
    round(time_25d, 4),
    "seconds"
)

print(
    "2.5D memory before:",
    round(memory_before_25d, 2),
    "MB"
)

print(
    "2.5D memory after:",
    round(memory_after_25d, 2),
    "MB"
)

print(
    "2.5D peak memory:",
    round(peak_25d, 2),
    "MB"
)


# ============================================================
# COMPARISON
# ============================================================

memory_saved = (
    peak_3d - peak_25d
)

time_difference = (
    time_3d - time_25d
)


print()
print(
    "============================================"
)

print(
    "FINAL COMPARISON"
)

print(
    "============================================"
)

print(
    f"3D mapping time: "
    f"{time_3d:.4f} s"
)

print(
    f"2.5D mapping time: "
    f"{time_25d:.4f} s"
)

print()

print(
    f"3D peak memory: "
    f"{peak_3d:.2f} MB"
)

print(
    f"2.5D peak memory: "
    f"{peak_25d:.2f} MB"
)

print()

print(
    f"Memory difference: "
    f"{memory_saved:.2f} MB"
)

print(
    f"Time difference: "
    f"{time_difference:.4f} s"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "mapping_comparison.csv",
    "w"
) as file:

    file.write(
        "metric,3D,2.5D\n"
    )

    file.write(
        f"mapping_time_seconds,"
        f"{time_3d},"
        f"{time_25d}\n"
    )

    file.write(
        f"peak_memory_MB,"
        f"{peak_3d},"
        f"{peak_25d}\n"
    )

    file.write(
        f"output_points_or_cells,"
        f"{points_3d},"
        f"{occupied_cells}\n"
    )


# ============================================================
# CLEANUP
# ============================================================

lidar.destroy()

vehicle.destroy()


print()
print(
    "Saved:"
)

print(
    "benchmark_input.ply"
)

print(
    "benchmark_3d_map.ply"
)

print(
    "benchmark_2_5d_height_map.npy"
)

print(
    "mapping_comparison.csv"
)

print()
print(
    "Benchmark complete."
)