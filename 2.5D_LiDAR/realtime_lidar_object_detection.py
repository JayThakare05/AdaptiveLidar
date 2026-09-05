import carla
import time
import random
import numpy as np
import open3d as o3d


# ============================================================
# CONFIG
# ============================================================

LIDAR_RANGE = 50.0

COLLECTION_TIME = 30.0

DBSCAN_EPS = 0.8

DBSCAN_MIN_POINTS = 8

MIN_CLUSTER_POINTS = 12

GROUND_DISTANCE_THRESHOLD = 0.15

MIN_OBJECT_HEIGHT = 0.25

MAX_OBJECT_HEIGHT = 4.0

MAX_OBJECT_LENGTH = 8.0

MAX_OBJECT_WIDTH = 5.0


# ============================================================
# CONNECT
# ============================================================

client = carla.Client(
    "localhost",
    2000
)

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

    print("ERROR: Tesla could not be spawned.")

    exit()


print(
    "Tesla spawned:",
    vehicle.id
)


# ============================================================
# TRAFFIC MANAGER
# ============================================================

traffic_manager = client.get_trafficmanager(
    8000
)

traffic_manager.set_global_distance_to_leading_vehicle(
    5.0
)

traffic_manager.global_percentage_speed_difference(
    20.0
)

# ============================================================
# CONTROLLED BACKGROUND TRAFFIC
# ============================================================

npc_vehicles = []

walkers = []

walker_controllers = []


NUM_BACKGROUND_VEHICLES = 12

NUM_PEDESTRIANS = 20


# ============================================================
# GET EGO LOCATION
# ============================================================

ego_location = vehicle.get_location()


# ============================================================
# BACKGROUND VEHICLES
# ============================================================

print()
print("Spawning nearby background vehicles...")


vehicle_spawn_points = (
    world.get_map().get_spawn_points()
)


# Sort spawn points by distance from ego
vehicle_spawn_points.sort(
    key=lambda sp:
        sp.location.distance(
            ego_location
        )
)


npc_blueprints = (
    blueprints.filter(
        "vehicle.*"
    )
)


for spawn_point in vehicle_spawn_points:

    if (
        len(npc_vehicles)
        >= NUM_BACKGROUND_VEHICLES
    ):
        break


    distance = (
        spawn_point.location.distance(
            ego_location
        )
    )


    # Keep NPCs reasonably close,
    # but don't spawn directly on top of Tesla.
    if distance < 12.0:
        continue


    if distance > 80.0:
        continue


    npc_bp = random.choice(
        npc_blueprints
    )


    npc = world.try_spawn_actor(
        npc_bp,
        spawn_point
    )


    if npc is None:
        continue


    npc.set_autopilot(
        True,
        traffic_manager.get_port()
    )


    npc_vehicles.append(
        npc
    )


print(
    "Nearby background vehicles:",
    len(npc_vehicles)
)


# ============================================================
# PEDESTRIANS
# ============================================================

print(
    "Spawning nearby pedestrians..."
)


walker_blueprints = (
    blueprints.filter(
        "walker.pedestrian.*"
    )
)


for i in range(
    NUM_PEDESTRIANS
):

    # Try multiple times to find a
    # navigation location near Tesla.
    spawn_location = None


    for attempt in range(20):

        candidate = (
            world.get_random_location_from_navigation()
        )


        if candidate is None:
            continue


        distance = (
            candidate.distance(
                ego_location
            )
        )


        if (
            10.0 <= distance <= 60.0
        ):

            spawn_location = candidate
            break


    if spawn_location is None:
        continue


    walker_bp = random.choice(
        walker_blueprints
    )


    walker_transform = carla.Transform(
        spawn_location
    )


    walker = world.try_spawn_actor(
        walker_bp,
        walker_transform
    )


    if walker is None:
        continue


    controller_bp = blueprints.find(
        "controller.ai.walker"
    )


    controller = world.spawn_actor(
        controller_bp,
        carla.Transform(),
        attach_to=walker
    )


    if controller is None:

        walker.destroy()

        continue


    walkers.append(
        walker
    )

    walker_controllers.append(
        controller
    )


print(
    "Nearby pedestrians:",
    len(walkers)
)


# ============================================================
# START PEDESTRIANS
# ============================================================

for controller in walker_controllers:

    controller.start()


    destination = None


    for attempt in range(20):

        candidate = (
            world.get_random_location_from_navigation()
        )


        if candidate is None:
            continue


        destination = candidate

        break


    if destination is not None:

        controller.go_to_location(
            destination
        )


    controller.set_max_speed(
        random.uniform(
            1.0,
            2.0
        )
    )

    print()
    print("============================================")
    print("TRAFFIC SANITY CHECK")
    print("============================================")

    print(
        "Ego Tesla ID:",
        vehicle.id
    )

    print(
        "Background vehicles:",
        len(npc_vehicles)
    )

    print(
        "Pedestrians:",
        len(walkers)
    )

    print(
        "Total actors:",
        len(npc_vehicles) +
        len(walkers) +
        1
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

# Capture at 10 Hz
lidar_bp.set_attribute(
    "sensor_tick",
    "0.1"
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
# CAMERA
# ============================================================

spectator = world.get_spectator()


# ============================================================
# DETECTION STORAGE
# ============================================================

latest_detection_cloud = None
latest_boxes = []

best_detection_cloud = None
best_detection_boxes = []
best_detection_count = 0

frame_count = 0


# ============================================================
# OBJECT CLASSIFICATION
# ============================================================

def classify_object(
    length,
    width,
    height
):

    # --------------------------------------------------------
    # VEHICLE-LIKE
    # --------------------------------------------------------

    if (
        2.0 <= length <= 6.0
        and
        1.2 <= width <= 2.8
        and
        0.8 <= height <= 2.5
    ):

        return "vehicle-like"


    # --------------------------------------------------------
    # PEDESTRIAN-LIKE
    # --------------------------------------------------------

    if (
        0.25 <= length <= 1.2
        and
        0.25 <= width <= 1.2
        and
        1.2 <= height <= 2.3
    ):

        return "pedestrian-like"


    # --------------------------------------------------------
    # POLE-LIKE
    # --------------------------------------------------------

    if (
        width <= 0.8
        and
        length <= 0.8
        and
        height >= 2.0
    ):

        return "pole-like"


    return "unknown"


# ============================================================
# DETECTION FUNCTION
# ============================================================

def detect_objects(
    xyz
):

    if len(xyz) < DBSCAN_MIN_POINTS:

        return [], None


    # ========================================================
    # CREATE CLOUD
    # ========================================================

    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            xyz
        )
    )


    # ========================================================
    # GROUND SEGMENTATION
    # ========================================================

    try:

        plane_model, inliers = (
            cloud.segment_plane(
                distance_threshold=
                GROUND_DISTANCE_THRESHOLD,

                ransac_n=3,

                num_iterations=300
            )
        )

    except RuntimeError:

        return [], cloud


    inliers = np.asarray(
        inliers
    )


    if len(inliers) == 0:

        return [], cloud


    # Remove ground
    non_ground_cloud = (
        cloud.select_by_index(
            inliers,
            invert=True
        )
    )


    non_ground = np.asarray(
        non_ground_cloud.points
    )


    if len(non_ground) < (
        DBSCAN_MIN_POINTS
    ):

        return [], non_ground_cloud


    # ========================================================
    # DBSCAN
    # ========================================================

    labels = np.array(
        non_ground_cloud.cluster_dbscan(

            eps=DBSCAN_EPS,

            min_points=DBSCAN_MIN_POINTS,

            print_progress=False
        )
    )


    cluster_count = (
        labels.max() + 1
    )


    objects = []


    # ========================================================
    # EXTRACT CLUSTERS
    # ========================================================

    for cluster_id in range(
        cluster_count
    ):

        indices = np.where(
            labels == cluster_id
        )[0]


        if len(indices) < (
            MIN_CLUSTER_POINTS
        ):

            continue


        cluster = non_ground[
            indices
        ]


        min_bound = (
            cluster.min(
                axis=0
            )
        )

        max_bound = (
            cluster.max(
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


        # ====================================================
        # OBJECT SIZE FILTER
        # ====================================================

        if height < MIN_OBJECT_HEIGHT:

            continue


        if height > MAX_OBJECT_HEIGHT:

            continue


        if length > MAX_OBJECT_LENGTH:

            continue


        if width > MAX_OBJECT_WIDTH:

            continue


        center = (
            min_bound +
            max_bound
        ) / 2.0


        object_type = classify_object(
            length,
            width,
            height
        )


        objects.append(
            {
                "cluster_id":
                    cluster_id,

                "points":
                    len(cluster),

                "center":
                    center,

                "dimensions":
                    dimensions,

                "min":
                    min_bound,

                "max":
                    max_bound,

                "type":
                    object_type
            }
        )


    return (
        objects,
        non_ground_cloud
    )


# ============================================================
# LiDAR CALLBACK
# ============================================================

def lidar_callback(data):

    global frame_count
    global best_detection_cloud
    global best_detection_boxes
    global best_detection_count
    global latest_detection_cloud
    global latest_boxes

    frame_count += 1


    # ========================================================
    # RAW DATA
    # ========================================================

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


    # ========================================================
    # VALIDATION
    # ========================================================

    valid = np.isfinite(
        xyz
    ).all(axis=1)

    xyz = xyz[valid]


    if len(xyz) == 0:

        return


    # ========================================================
    # RANGE FILTER
    # ========================================================

    distances = np.sqrt(
        np.sum(
            xyz ** 2,
            axis=1
        )
    )


    valid = (
        distances <= LIDAR_RANGE
    )


    xyz = xyz[valid]


    # ========================================================
    # DETECTION
    # ========================================================

    objects, cloud = detect_objects(
        xyz
    )


    latest_detection_cloud = cloud
    latest_boxes = objects


    # Keep the best frame encountered during the run.
    # Prefer frames with the most detected objects.
    if (
        cloud is not None
        and
        len(cloud.points) > 0
        and
        len(objects) > best_detection_count
    ):

        best_detection_cloud = cloud
        best_detection_boxes = objects
        best_detection_count = len(objects)

        print(
            f"*** New best detection frame: "
            f"{data.frame} "
            f"with {len(objects)} objects ***"
        )



    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        "--------------------------------------------"
    )

    print(
        "LiDAR Frame:",
        data.frame
    )

    print(
        "Raw points:",
        len(raw)
    )

    print(
        "Usable points:",
        len(xyz)
    )

    print(
        "Detected objects:",
        len(objects)
    )


    for i, obj in enumerate(
        objects
    ):

        center = obj["center"]

        dimensions = (
            obj["dimensions"]
        )


        print(
            f"  Object {i + 1}: "
            f"{obj['type']}"
        )

        print(
            f"     Points: "
            f"{obj['points']}"
        )

        print(
            f"     Position: "
            f"("
            f"{center[0]:.2f}, "
            f"{center[1]:.2f}, "
            f"{center[2]:.2f}"
            f")"
        )

        print(
            f"     Size: "
            f"{dimensions[0]:.2f} x "
            f"{dimensions[1]:.2f} x "
            f"{dimensions[2]:.2f} m"
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
print(
    "============================================"
)

print(
    "REAL-TIME LiDAR OBJECT DETECTION"
)

print(
    "============================================"
)

print()
print(
    "Tesla is driving."
)

print(
    "Detection runs on individual LiDAR frames."
)

print()


# ============================================================
# DRIVE
# ============================================================

start_time = time.time()


while (
    time.time() - start_time
    < COLLECTION_TIME
):


    # --------------------------------------------------------
    # FOLLOWING CAMERA
    # --------------------------------------------------------

    vehicle_transform = (
        vehicle.get_transform()
    )


    camera_location = (
        vehicle_transform.transform(
            carla.Location(
                x=-14.0,
                y=0.0,
                z=4.0
            )
        )
    )


    # Face in same direction as Tesla
    camera_rotation = carla.Rotation(
        pitch=-15.0,

        yaw=(
            vehicle_transform.rotation.yaw
        ),

        roll=0.0
    )


    spectator.set_transform(
        carla.Transform(
            camera_location,
            camera_rotation
        )
    )


    time.sleep(
        0.10
    )


# ============================================================
# STOP
# ============================================================

lidar.stop()

time.sleep(
    0.5
)


# ============================================================
# FINAL FRAME VISUALIZATION
# ============================================================

print()
print(
    "============================================"
)

print(
    "FINAL DETECTION FRAME"
)

print(
    "============================================"
)


# ============================================================
# BEST DETECTION VISUALIZATION
# ============================================================

print()
print(
    "============================================"
)

print(
    "BEST DETECTION FRAME"
)

print(
    "============================================"
)


if (
    best_detection_cloud is not None
    and
    len(best_detection_cloud.points) > 0
):

    print(
        "Best detected objects:",
        best_detection_count
    )

    print(
        "Best cloud points:",
        len(best_detection_cloud.points)
    )


    # --------------------------------------------------------
    # COLOR POINT CLOUD
    # --------------------------------------------------------

    best_detection_cloud.paint_uniform_color(
        [0.1, 0.8, 1.0]
    )


    geometries = []

    geometries.append(
        best_detection_cloud
    )


    # --------------------------------------------------------
    # DRAW BOUNDING BOXES
    # --------------------------------------------------------

    for obj in best_detection_boxes:

        box = (
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=obj["min"],
                max_bound=obj["max"]
            )
        )

        box.color = (
            1.0,
            0.0,
            0.0
        )

        geometries.append(
            box
        )


    # --------------------------------------------------------
    # ADD COORDINATE FRAME
    # --------------------------------------------------------

    frame = (
        o3d.geometry.TriangleMesh
        .create_coordinate_frame(
            size=2.0
        )
    )

    geometries.append(
        frame
    )


    print(
        "Opening best detection visualization..."
    )


    o3d.visualization.draw_geometries(
        geometries,
        window_name=(
            "REAL-TIME LiDAR "
            "OBJECT DETECTION"
        )
    )

else:

    print(
        "No usable detection frame "
        "was captured."
    )


# ============================================================
# CLEANUP
# ============================================================

# Stop and destroy LiDAR
lidar.destroy()


# Stop and destroy pedestrian controllers
for controller in walker_controllers:

    try:
        controller.stop()
    except:
        pass

    controller.destroy()


# Destroy pedestrians
for walker in walkers:
    walker.destroy()


# Destroy background vehicles
for npc in npc_vehicles:
    npc.destroy()


# Destroy ego Tesla
vehicle.destroy()


print()
print(
    "All vehicles and pedestrians destroyed."
)

print(
    "Detection experiment complete."
)