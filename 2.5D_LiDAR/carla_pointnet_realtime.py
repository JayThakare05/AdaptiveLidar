import os
import random
import time
import threading
import csv

import numpy as np
import torch
import open3d as o3d
import carla

from train_pointnet2 import PointNet2Segmentation


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "pointnet2_best.pth"

NUM_CLASSES = 7
NUM_POINTS = 2048

COLLECTION_TIME = 30.0

NUM_BACKGROUND_VEHICLES = 10
NUM_PEDESTRIANS = 15

# ============================================================
# LiDAR CONFIGURATION
# ============================================================

LIDAR_RANGE = 50.0
LIDAR_CHANNELS = 32

LIDAR_POINTS_PER_SECOND = 130000
LIDAR_ROTATION_FREQUENCY = 10.0
LIDAR_SENSOR_TICK = 0.1

# ============================================================
# OBJECT CLUSTERING
# ============================================================

DBSCAN_EPS = 1.5
DBSCAN_MIN_POINTS = 8


# ============================================================
# SEMANTIC CLASSES
# ============================================================

CLASS_NAMES = {
    0: "Drivable",
    1: "Non-drivable",
    2: "Static obstacle",
    3: "Dynamic object",
    4: "Vegetation",
    5: "Road marking",
    6: "Other"
}


# ============================================================
# COLORS
# ============================================================

CLASS_COLORS = {
    0: [0.35, 0.35, 0.35],
    1: [0.65, 0.65, 0.65],
    2: [1.00, 0.10, 0.10],
    3: [1.00, 0.50, 0.00],
    4: [0.10, 0.70, 0.10],
    5: [1.00, 0.85, 0.00],
    6: [0.60, 0.10, 0.70],
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 70)
print("CARLA + POINTNET++ LIVE SEMANTIC DETECTION")
print("=" * 70)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n"
        f"{os.path.abspath(MODEL_PATH)}"
    )


print()
print("Creating PointNet++ model...")

model = PointNet2Segmentation().to(
    DEVICE
)


print(
    "Loading:",
    os.path.abspath(MODEL_PATH)
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# MODEL WARM-UP + TIMING
# ============================================================

print()
print(
    "============================================"
)
print(
    "POINTNET++ WARM-UP"
)
print(
    "============================================"
)


warmup_input = torch.randn(
    1,
    NUM_POINTS,
    3,
    dtype=torch.float32,
    device=DEVICE
)


warmup_times = []


for i in range(3):

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    start = time.perf_counter()


    with torch.inference_mode():

        _ = model(
            warmup_input
        )


    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    elapsed = (
        time.perf_counter()
        -
        start
    )


    warmup_times.append(
        elapsed
    )


    print(
        f"Warm-up {i + 1}: "
        f"{elapsed:.4f}s"
    )


del warmup_input


print()

print(
    "Steady warm-up time:",
    f"{warmup_times[-1]:.4f}s"
)

print(
    "PointNet++ warm-up complete."
)


# ============================================================
# CONNECT TO CARLA
# ============================================================

client = carla.Client(
    "localhost",
    2000
)

client.set_timeout(
    10.0
)

world = client.get_world()

blueprints = (
    world.get_blueprint_library()
)


print()
print(
    "CARLA map:",
    world.get_map().name
)


# ============================================================
# ASYNC MODE
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
# SPAWN EGO
# ============================================================

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


for sp in spawn_points:

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        sp
    )

    if vehicle is not None:

        break


if vehicle is None:

    raise RuntimeError(
        "Could not spawn Tesla."
    )


print(
    "Tesla spawned:",
    vehicle.id
)


# ============================================================
# TRAFFIC MANAGER
# ============================================================

traffic_manager = (
    client.get_trafficmanager(
        8000
    )
)


traffic_manager.set_global_distance_to_leading_vehicle(
    8.0
)


traffic_manager.global_percentage_speed_difference(
    10.0
)


# ============================================================
# BACKGROUND VEHICLES
# ============================================================

npc_vehicles = []

ego_location = (
    vehicle.get_location()
)

spawn_points = (
    world.get_map().get_spawn_points()
)


safe_spawns = []


for sp in spawn_points:

    distance = (
        sp.location.distance(
            ego_location
        )
    )


    if (
        25.0
        <= distance
        <= 100.0
    ):

        safe_spawns.append(
            sp
        )


random.shuffle(
    safe_spawns
)


npc_blueprints = (
    blueprints.filter(
        "vehicle.*"
    )
)


print()
print(
    "Spawning background vehicles..."
)


for sp in safe_spawns:

    if (
        len(npc_vehicles)
        >= NUM_BACKGROUND_VEHICLES
    ):

        break


    npc_bp = random.choice(
        npc_blueprints
    )


    npc = world.try_spawn_actor(
        npc_bp,
        sp
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
    "Background vehicles:",
    len(npc_vehicles)
)


# ============================================================
# PEDESTRIANS
# ============================================================

walkers = []
walker_controllers = []


walker_blueprints = (
    blueprints.filter(
        "walker.pedestrian.*"
    )
)


walker_controller_bp = (
    blueprints.find(
        "controller.ai.walker"
    )
)


print(
    "Spawning pedestrians..."
)


for _ in range(
    NUM_PEDESTRIANS
):

    location = None


    for _ in range(40):

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
            15.0
            <= distance
            <= 70.0
        ):

            location = candidate

            break


    if location is None:

        continue


    walker_bp = random.choice(
        walker_blueprints
    )


    walker = world.try_spawn_actor(
        walker_bp,
        carla.Transform(
            location
        )
    )


    if walker is None:

        continue


    controller = world.spawn_actor(
        walker_controller_bp,
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
    "Pedestrians:",
    len(walkers)
)


# ============================================================
# START PEDESTRIANS
# ============================================================

for controller in walker_controllers:

    controller.start()


    destination = (
        world.get_random_location_from_navigation()
    )


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


# ============================================================
# TRAFFIC CHECK
# ============================================================

print()
print("=" * 50)
print("TRAFFIC SANITY CHECK")
print("=" * 50)

print(
    "Ego Tesla:",
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
    "Total traffic actors:",
    1 +
    len(npc_vehicles) +
    len(walkers)
)


# ============================================================
# LiDAR
# ============================================================

lidar_bp = blueprints.find(
    "sensor.lidar.ray_cast"
)


lidar_bp.set_attribute(
    "channels",
    str(LIDAR_CHANNELS)
)


lidar_bp.set_attribute(
    "range",
    str(LIDAR_RANGE)
)


lidar_bp.set_attribute(
    "points_per_second",
    str(LIDAR_POINTS_PER_SECOND)
)


lidar_bp.set_attribute(
    "rotation_frequency",
    str(LIDAR_ROTATION_FREQUENCY)
)


lidar_bp.set_attribute(
    "sensor_tick",
    str(LIDAR_SENSOR_TICK)
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


print()
print(
    "LiDAR attached:",
    lidar.id
)

print(
    f"LiDAR: "
    f"{LIDAR_CHANNELS} channels | "
    f"{LIDAR_POINTS_PER_SECOND} pts/s | "
    f"{LIDAR_SENSOR_TICK}s tick"
)


# ============================================================
# SHARED LIDAR DATA
# ============================================================

latest_xyz = None
latest_frame_id = None
frames_received = 0

lidar_lock = threading.Lock()


# ============================================================
# LiDAR CALLBACK
# ============================================================

def lidar_callback(data):

    global latest_xyz
    global latest_frame_id
    global frames_received


    raw = np.frombuffer(
        data.raw_data,
        dtype=np.float32
    )


    raw = raw.reshape(
        (-1, 4)
    )


    xyz = raw[:, :3].astype(
        np.float32
    )


    valid = np.isfinite(
        xyz
    ).all(axis=1)


    xyz = xyz[
        valid
    ]


    if len(xyz) == 0:

        return


    distance = np.linalg.norm(
        xyz,
        axis=1
    )


    xyz = xyz[
        distance <= LIDAR_RANGE
    ]


    if len(xyz) == 0:

        return


    with lidar_lock:

        latest_xyz = xyz.copy()

        latest_frame_id = data.frame

        frames_received += 1


# ============================================================
# CAMERA THREAD
#
# Camera runs independently of PointNet.
# ============================================================

camera_stop = threading.Event()


def camera_worker():

    spectator = (
        world.get_spectator()
    )


    while not camera_stop.is_set():

        try:

            transform = (
                vehicle.get_transform()
            )


            camera_location = (
                transform.transform(
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
                    yaw=transform.rotation.yaw,
                    roll=0.0
                )
            )


            spectator.set_transform(
                carla.Transform(
                    camera_location,
                    camera_rotation
                )
            )


        except Exception:

            pass


        time.sleep(
            0.02
        )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_for_pointnet(
    xyz
):

    n = len(xyz)


    if n == 0:

        return None


    rng = np.random.default_rng()


    if n >= NUM_POINTS:

        indices = rng.choice(
            n,
            NUM_POINTS,
            replace=False
        )

    else:

        indices = rng.choice(
            n,
            NUM_POINTS,
            replace=True
        )


    sampled_raw = (
        xyz[
            indices
        ].astype(
            np.float32
        )
    )


    centroid = np.mean(
        sampled_raw,
        axis=0,
        keepdims=True
    )


    normalized = (
        sampled_raw -
        centroid
    )


    scale = np.max(
        np.linalg.norm(
            normalized,
            axis=1
        )
    )


    if scale > 0:

        normalized = (
            normalized /
            scale
        )


    return (
        sampled_raw,
        normalized
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_pointnet(
    xyz
):

    result = preprocess_for_pointnet(
        xyz
    )


    if result is None:

        return None


    sampled_raw = result[0]

    normalized = result[1]


    input_tensor = (
        torch.from_numpy(
            normalized
        )
        .unsqueeze(0)
        .to(
            DEVICE,
            non_blocking=True
        )
    )


    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    start = time.perf_counter()


    with torch.inference_mode():

        logits = model(
            input_tensor
        )


        predictions = (
            torch.argmax(
                logits,
                dim=-1
            )
        )


    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    inference_time = (
        time.perf_counter()
        -
        start
    )


    predictions = (
        predictions
        .squeeze(0)
        .cpu()
        .numpy()
    )


    return {
        "points":
            sampled_raw,

        "predictions":
            predictions,

        "inference_time":
            inference_time
    }


# ============================================================
# DYNAMIC OBJECT DETECTION
# ============================================================

def detect_dynamic_objects(
    points,
    predictions
):

    dynamic_points = (
        points[
            predictions == 3
        ]
    )


    if len(dynamic_points) < (
        DBSCAN_MIN_POINTS
    ):

        return []


    cloud = (
        o3d.geometry.PointCloud()
    )


    cloud.points = (
        o3d.utility.Vector3dVector(
            dynamic_points
        )
    )


    cluster_labels = (
        np.asarray(
            cloud.cluster_dbscan(
                eps=DBSCAN_EPS,
                min_points=DBSCAN_MIN_POINTS,
                print_progress=False
            )
        )
    )


    objects = []


    for cluster_id in sorted(
        set(cluster_labels)
    ):

        if cluster_id < 0:

            continue


        cluster_points = (
            dynamic_points[
                cluster_labels ==
                cluster_id
            ]
        )


        if len(cluster_points) < (
            DBSCAN_MIN_POINTS
        ):

            continue


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


        # Reject tiny noise clusters.

        if (
            dimensions[0] < 0.15
            and
            dimensions[1] < 0.15
        ):

            continue


        center = (
            min_bound +
            max_bound
        ) / 2.0


        objects.append(
            {
                "min":
                    min_bound,

                "max":
                    max_bound,

                "center":
                    center,

                "dimensions":
                    dimensions,

                "points":
                    len(cluster_points)
            }
        )


    return objects


# ============================================================
# START LiDAR
# ============================================================

lidar.listen(
    lidar_callback
)


# ============================================================
# START CAMERA THREAD
# ============================================================

camera_thread = threading.Thread(
    target=camera_worker,
    daemon=True
)

camera_thread.start()


# ============================================================
# START EGO AUTOPILOT
# ============================================================

vehicle.set_autopilot(
    True,
    traffic_manager.get_port()
)


print()
print("=" * 70)
print("CARLA LIVE POINTNET++ TEST")
print("=" * 70)

print(
    "Tesla is driving."
)

print(
    "Camera follows Tesla in a separate thread."
)

print(
    "LiDAR callback only stores frames."
)

print(
    "PointNet++ inference runs in main thread."
)

print(
    f"Test duration: {COLLECTION_TIME}s"
)

print()


# ============================================================
# MAIN INFERENCE LOOP
# ============================================================

start_time = time.time()

last_processed_frame = None

inference_count = 0

inference_times = []

best_result = None
best_dynamic_points = -1

last_status = time.time()


try:

    while (
        time.time()
        -
        start_time
        <
        COLLECTION_TIME
    ):

        # ----------------------------------------------------
        # Get newest LiDAR frame
        # ----------------------------------------------------

        with lidar_lock:

            if latest_xyz is None:

                current_xyz = None
                current_frame = None

            else:

                current_xyz = (
                    latest_xyz.copy()
                )

                current_frame = (
                    latest_frame_id
                )


        if (
            current_xyz is not None
            and
            current_frame is not None
            and
            current_frame !=
            last_processed_frame
        ):

            last_processed_frame = (
                current_frame
            )


            # ------------------------------------------------
            # INFERENCE
            # ------------------------------------------------

            result = run_pointnet(
                current_xyz
            )


            if result is not None:

                inference_count += 1

                points = result[
                    "points"
                ]

                predictions = result[
                    "predictions"
                ]

                inference_time = result[
                    "inference_time"
                ]


                inference_times.append(
                    inference_time
                )


                objects = (
                    detect_dynamic_objects(
                        points,
                        predictions
                    )
                )


                dynamic_points = int(
                    np.sum(
                        predictions == 3
                    )
                )


                # ------------------------------------------------
                # BEST RESULT
                # ------------------------------------------------

                if (
                    dynamic_points >
                    best_dynamic_points
                ):

                    best_dynamic_points = (
                        dynamic_points
                    )

                    best_result = {
                        "frame":
                            current_frame,

                        "points":
                            points.copy(),

                        "predictions":
                            predictions.copy(),

                        "objects":
                            list(objects),

                        "inference_time":
                            inference_time
                    }


                # ------------------------------------------------
                # PRINT
                # ------------------------------------------------

                counts = np.bincount(
                    predictions,
                    minlength=NUM_CLASSES
                )


                print()
                print(
                    "--------------------------------------------"
                )

                print(
                    "Processed frame:",
                    current_frame
                )

                print(
                    "Raw LiDAR points:",
                    len(current_xyz)
                )

                print(
                    "PointNet input:",
                    NUM_POINTS
                )

                print(
                    "Inference time:",
                    f"{inference_time:.4f}s"
                )

                print(
                    "Dynamic points:",
                    dynamic_points
                )

                print(
                    "Dynamic clusters:",
                    len(objects)
                )


                print(
                    "Semantic classes:"
                )


                for class_id in range(
                    NUM_CLASSES
                ):

                    if counts[class_id] > 0:

                        print(
                            f"  "
                            f"{class_id}: "
                            f"{CLASS_NAMES[class_id]:<18}"
                            f"→ "
                            f"{counts[class_id]}"
                        )


                for i, obj in enumerate(
                    objects
                ):

                    c = obj[
                        "center"
                    ]

                    d = obj[
                        "dimensions"
                    ]


                    print(
                        f"  Dynamic Object "
                        f"{i + 1}: "
                        f"center=("
                        f"{c[0]:.2f}, "
                        f"{c[1]:.2f}, "
                        f"{c[2]:.2f}) "
                        f"size=("
                        f"{d[0]:.2f} × "
                        f"{d[1]:.2f} × "
                        f"{d[2]:.2f}) "
                        f"points="
                        f"{obj['points']}"
                    )


        # --------------------------------------------------------
        # STATUS
        # --------------------------------------------------------

        if (
            time.time()
            -
            last_status
            >= 5.0
        ):

            print()
            print(
                "STATUS"
            )

            print(
                "  LiDAR frames received:",
                frames_received
            )

            print(
                "  Inference frames:",
                inference_count
            )


            if inference_times:

                print(
                    "  Avg inference:",
                    f"{np.mean(inference_times):.4f}s"
                )


            last_status = time.time()


        time.sleep(
            0.005
        )


finally:

    camera_stop.set()

    try:

        lidar.stop()

    except Exception:

        pass


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    "LiDAR frames received:",
    frames_received
)

print(
    "Inference frames:",
    inference_count
)


if inference_times:

    print(
        "Average inference:",
        f"{np.mean(inference_times):.4f}s"
    )

    print(
        "Fastest inference:",
        f"{np.min(inference_times):.4f}s"
    )

    print(
        "Slowest inference:",
        f"{np.max(inference_times):.4f}s"
    )


# ============================================================
# FINAL VISUALIZATION
# ============================================================

if best_result is None:

    print()
    print(
        "WARNING: PointNet++ did not finish an inference."
    )

    print(
        "No Open3D visualization available."
    )

else:

    points = best_result[
        "points"
    ]

    predictions = best_result[
        "predictions"
    ]

    objects = best_result[
        "objects"
    ]


    print()
    print(
        "Best frame:",
        best_result["frame"]
    )

    print(
        "Best dynamic points:",
        int(
            np.sum(
                predictions == 3
            )
        )
    )

    # --------------------------------------------------------
    # COLORS
    # --------------------------------------------------------

    colors = np.zeros(
        (
            len(points),
            3
        ),
        dtype=np.float64
    )


    for class_id, color in (
        CLASS_COLORS.items()
    ):

        mask = (
            predictions ==
            class_id
        )

        colors[
            mask
        ] = color


    # --------------------------------------------------------
    # CLOUD
    # --------------------------------------------------------

    cloud = (
        o3d.geometry.PointCloud()
    )


    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )


    cloud.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )


    geometries = [
        cloud
    ]


    # --------------------------------------------------------
    # BOXES
    # --------------------------------------------------------

    for obj in objects:

        box = (
            o3d.geometry
            .AxisAlignedBoundingBox(
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
    # SMALL COORDINATE FRAME
    # --------------------------------------------------------

    frame = (
        o3d.geometry
        .TriangleMesh
        .create_coordinate_frame(
            size=1.0
        )
    )


    geometries.append(
        frame
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    o3d.io.write_point_cloud(
        "carla_pointnet_realtime_best.ply",
        cloud
    )


    with open(
        "carla_pointnet_realtime_detections.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "object_id",
                "points",
                "center_x",
                "center_y",
                "center_z",
                "length",
                "width",
                "height"
            ]
        )


        for i, obj in enumerate(
            objects
        ):

            c = obj[
                "center"
            ]

            d = obj[
                "dimensions"
            ]


            writer.writerow(
                [
                    i + 1,
                    obj["points"],
                    c[0],
                    c[1],
                    c[2],
                    d[0],
                    d[1],
                    d[2]
                ]
            )


    print()
    print(
        "Opening Open3D visualization..."
    )


    o3d.visualization.draw_geometries(
        geometries,
        window_name=(
            "CARLA + PointNet++ "
            "REAL LiDAR SEMANTIC DETECTION"
        )
    )


# ============================================================
# CLEANUP
# ============================================================

try:

    lidar.destroy()

except Exception:

    pass


for controller in walker_controllers:

    try:

        controller.stop()

    except Exception:

        pass


    try:

        controller.destroy()

    except Exception:

        pass


for walker in walkers:

    try:

        walker.destroy()

    except Exception:

        pass


for npc in npc_vehicles:

    try:

        npc.destroy()

    except Exception:

        pass


try:

    vehicle.destroy()

except Exception:

    pass


print()
print(
    "All CARLA actors cleaned up."
)

print(
    "Experiment complete."
)