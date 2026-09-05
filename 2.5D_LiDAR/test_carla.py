import carla
import time
import random
import numpy as np
import open3d as o3d


# ============================================================
# 1. CONNECT TO CARLA
# ============================================================

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprints = world.get_blueprint_library()

print("Connected to CARLA")
print("Map:", world.get_map().name)


# ============================================================
# 2. MAKE SURE CARLA IS IN ASYNCHRONOUS MODE
# ============================================================

settings = world.get_settings()

if settings.synchronous_mode:

    print("CARLA is currently in synchronous mode.")
    print("Switching to asynchronous mode...")

    settings.synchronous_mode = False
    world.apply_settings(settings)

    print("Asynchronous mode enabled.")

else:

    print("CARLA is already in asynchronous mode.")


# ============================================================
# 3. SPAWN TESLA
# ============================================================

vehicle_bp = blueprints.find("vehicle.tesla.model3")

spawn_points = world.get_map().get_spawn_points()

# Randomize spawn points so we don't always try the same
# location if another vehicle is already there.
random.shuffle(spawn_points)

vehicle = None

for spawn_point in spawn_points:

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_point
    )

    if vehicle is not None:
        break


if vehicle is None:

    print("Could not spawn Tesla.")
    print("All available spawn points may be occupied.")

    exit()


print("Tesla spawned:", vehicle.id)


# ============================================================
# 4. CREATE LiDAR
# ============================================================

lidar_bp = blueprints.find(
    "sensor.lidar.ray_cast"
)

# LiDAR configuration
lidar_bp.set_attribute(
    "channels",
    "32"
)

lidar_bp.set_attribute(
    "range",
    "50"
)

lidar_bp.set_attribute(
    "points_per_second",
    "56000"
)

lidar_bp.set_attribute(
    "rotation_frequency",
    "10"
)


# LiDAR mounted 2.5 m above vehicle origin
lidar_transform = carla.Transform(
    carla.Location(
        x=0,
        y=0,
        z=2.5
    )
)


lidar = world.spawn_actor(
    lidar_bp,
    lidar_transform,
    attach_to=vehicle
)

print("LiDAR attached:", lidar.id)


# ============================================================
# 5. SAVE INITIAL LiDAR WORLD POSITION
# ============================================================

initial_lidar_transform = lidar.get_transform()

initial_lidar_position = (
    initial_lidar_transform.location
)

initial_lidar_origin = np.array([
    initial_lidar_position.x,
    initial_lidar_position.y,
    initial_lidar_position.z
])

# Convert CARLA coordinate system to the convention
# we will use for Open3D.
initial_lidar_origin[1] *= -1


# ============================================================
# 6. LiDAR MARKER IN CARLA
# ============================================================

world.debug.draw_point(
    initial_lidar_position,
    size=0.25,
    life_time=60.0
)

print("LiDAR marker placed in CARLA.")


# ============================================================
# 7. MOVE CARLA CAMERA BEHIND TESLA
# ============================================================

spectator = world.get_spectator()

vehicle_transform = vehicle.get_transform()

camera_location = vehicle_transform.transform(
    carla.Location(
        x=-8,
        y=0,
        z=4
    )
)

spectator.set_transform(
    carla.Transform(
        camera_location,
        vehicle_transform.rotation
    )
)


# ============================================================
# 8. STORAGE FOR LiDAR FRAMES
# ============================================================

all_points = []

COLLECTION_TIME = 5.0


# ============================================================
# 9. LiDAR CALLBACK
# ============================================================

def lidar_callback(data):

    # --------------------------------------------------------
    # Convert raw LiDAR bytes to float32
    # --------------------------------------------------------

    points = np.frombuffer(
        data.raw_data,
        dtype=np.float32
    )

    # Every LiDAR point contains:
    #
    # X
    # Y
    # Z
    # Intensity

    points = points.reshape((-1, 4))


    # --------------------------------------------------------
    # Extract XYZ
    # --------------------------------------------------------

    xyz = points[:, :3]


    # --------------------------------------------------------
    # Get LiDAR transformation matrix
    # --------------------------------------------------------

    transform = data.transform

    transformation_matrix = np.array(
        transform.get_matrix()
    )


    # --------------------------------------------------------
    # Convert XYZ → homogeneous coordinates
    # --------------------------------------------------------

    homogeneous_points = np.hstack(
        (
            xyz,
            np.ones(
                (xyz.shape[0], 1)
            )
        )
    )


    # --------------------------------------------------------
    # Transform LiDAR coordinates → WORLD coordinates
    # --------------------------------------------------------

    world_points = (
        transformation_matrix
        @ homogeneous_points.T
    ).T


    world_xyz = world_points[:, :3]


    # --------------------------------------------------------
    # CARLA → Open3D coordinate conversion
    #
    # Flip Y axis for right-handed visualization
    # --------------------------------------------------------

    world_xyz[:, 1] *= -1


    # --------------------------------------------------------
    # Store this frame
    # --------------------------------------------------------

    all_points.append(world_xyz)


    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    print(
        f"Frame {data.frame}: "
        f"{len(world_xyz)} points"
    )


# ============================================================
# 10. START LiDAR
# ============================================================

lidar.listen(lidar_callback)

print()
print("============================================")
print("LiDAR collection started")
print("============================================")
print(
    f"Collection time: {COLLECTION_TIME} seconds"
)
print()


# ============================================================
# 11. START TESLA AUTOPILOT
# ============================================================

vehicle.set_autopilot(True)

print("Tesla autopilot enabled.")
print()


# ============================================================
# 12. COLLECT LiDAR DATA
# ============================================================

collection_start = time.time()

while (
    time.time() - collection_start
    < COLLECTION_TIME
):

    time.sleep(0.1)


# ============================================================
# 13. STOP LiDAR
# ============================================================

lidar.stop()

print()
print("============================================")
print("LiDAR collection finished")
print("============================================")
print()


# ============================================================
# 14. CHECK WHETHER DATA WAS RECEIVED
# ============================================================

if len(all_points) == 0:

    print("No LiDAR data received.")

else:

    # ========================================================
    # 15. COMBINE ALL LiDAR FRAMES
    # ========================================================

    accumulated_points = np.vstack(
        all_points
    )

    print(
        "Total raw accumulated points:",
        len(accumulated_points)
    )


    # ========================================================
    # 16. CREATE OPEN3D POINT CLOUD
    # ========================================================

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        accumulated_points
    )


    # ========================================================
    # 17. VOXEL DOWNSAMPLING
    # ========================================================

    print()
    print("Applying voxel downsampling...")

    downsampled_pcd = pcd.voxel_down_sample(
        voxel_size=0.08
    )

    print(
        "Points after voxel downsampling:",
        len(downsampled_pcd.points)
    )


    # ========================================================
    # 18. STATISTICAL OUTLIER REMOVAL
    # ========================================================

    print()
    print("Removing statistical outliers...")

    if len(downsampled_pcd.points) >= 20:

        filtered_pcd, inlier_indices = (
            downsampled_pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )
        )

    else:

        print(
            "Not enough points for statistical "
            "outlier removal."
        )

        filtered_pcd = downsampled_pcd


    print(
        "Points after filtering:",
        len(filtered_pcd.points)
    )


    # ========================================================
    # 19. CREATE LiDAR ORIGIN MARKER IN OPEN3D
    # ========================================================

    coordinate_frame = (
        o3d.geometry.TriangleMesh
        .create_coordinate_frame(
            size=2.0,
            origin=initial_lidar_origin
        )
    )


    # ========================================================
    # 20. SAVE POINT CLOUD
    # ========================================================

    filename = "carla_lidar_map.ply"

    success = o3d.io.write_point_cloud(
        filename,
        filtered_pcd
    )

    print()

    if success:

        print(
            "Point cloud successfully saved:"
        )

        print(filename)

    else:

        print(
            "Failed to save point cloud."
        )


    # ========================================================
    # 21. OPEN3D VISUALIZATION
    # ========================================================

    print()
    print("============================================")
    print("Opening Open3D LiDAR MAP")
    print("============================================")
    print()
    print("Mouse controls:")
    print("  Left mouse   -> Rotate")
    print("  Scroll       -> Zoom")
    print("  Right mouse  -> Pan")
    print("  H            -> Help")
    print()
    print("Close the Open3D window when finished.")
    print()


    o3d.visualization.draw_geometries(
        [
            filtered_pcd,
            coordinate_frame
        ],
        window_name="CARLA LiDAR MAP"
    )


# ============================================================
# 22. CLEANUP
# ============================================================

print()
print("Cleaning up...")


try:
    lidar.stop()
except Exception:
    pass


try:
    lidar.destroy()
except Exception:
    pass


try:
    vehicle.destroy()
except Exception:
    pass


print("Tesla destroyed.")
print("LiDAR destroyed.")
print("Done.")