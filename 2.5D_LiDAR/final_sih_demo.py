import os
import random
import time
import threading
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import carla

from train_pointnet2 import PointNet2Segmentation


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "pointnet2_best.pth"

COLLECTION_TIME = 30.0

NUM_BACKGROUND_VEHICLES = 14
NUM_PEDESTRIANS = 18

NUM_POINTS = 2048
NUM_CLASSES = 7


# ============================================================
# POINTNET SETTINGS
# ============================================================

# Run PointNet periodically rather than on every LiDAR frame.
# This keeps the CARLA demonstration smooth.

POINTNET_INTERVAL = 1.5


# ============================================================
# LiDAR SETTINGS
# ============================================================

LIDAR_RANGE = 50.0

LIDAR_CHANNELS = 32

LIDAR_POINTS_PER_SECOND = 130000

LIDAR_ROTATION_FREQUENCY = 10.0

LIDAR_SENSOR_TICK = 0.1


# ============================================================
# ADAPTIVE 2.5D SETTINGS
# ============================================================

BASE_CELL_SIZE = 2.0

MEDIUM_CELL_SIZE = 1.0

FINE_CELL_SIZE = 0.5

# ============================================================
# DISPLAY SETTINGS
# ============================================================

# CARLA world Y is mirrored relative to the intuitive
# top-down view we want for the demo.
#
# IMPORTANT:
# This changes ONLY the visualization.
# The underlying world-coordinate data remains untouched.

DISPLAY_MIRROR_Y = True


# Minimum semantic evidence before displaying a cell.

MIN_CELL_POINTS = 2


# Only show the most recent dynamic-object predictions
# so moving vehicles/pedestrians do not leave long trails.

DYNAMIC_DISPLAY_SECONDS = 5.0

# Static environment can accumulate for the whole run.
STATIC_HISTORY_SECONDS = 30.0

# Moving objects should not leave a huge trail.
DYNAMIC_HISTORY_SECONDS = 5.0


# ============================================================
# DYNAMIC OBJECT CLUSTERING
# ============================================================

DBSCAN_EPS = 2.2

DBSCAN_MIN_POINTS = 6

# ============================================================
# DYNAMIC ACTOR VISUALIZATION (CARLA DEMO SUPPORT)
# ============================================================
# These settings are ONLY for making clearly moving simulated actors
# line up with the orange "Dynamic object" visualization.
# PointNet++ semantic predictions remain separate and unchanged.
DYNAMIC_ACTOR_MIN_SPEED = 0.5       # m/s
DYNAMIC_ACTOR_MAX_DISTANCE = 50.0   # LiDAR range
DYNAMIC_ACTOR_BOX_MARGIN = 0.35     # meters
DYNAMIC_ACTOR_LIDAR_SUPPORT_RADIUS = 2.5


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
# SEMANTIC COLORS
# ============================================================

CLASS_COLORS = {

    0: "#777777",

    1: "#BDBDBD",

    2: "#E53935",

    3: "#FB8C00",

    4: "#43A047",

    5: "#FDD835",

    6: "#8E44AD"
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
print("=" * 78)
print("FINAL SIH SEMANTIC LiDAR + ADAPTIVE 2.5D DEMO")
print("=" * 78)

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
# LOAD POINTNET++
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"PointNet++ model not found:\n"
        f"{os.path.abspath(MODEL_PATH)}"
    )


print()
print(
    "Creating PointNet++ model..."
)


model = (
    PointNet2Segmentation()
    .to(DEVICE)
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


if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

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
    "PointNet++ model loaded."
)


# ============================================================
# MODEL WARM-UP
# ============================================================

print()
print(
    "Warming up PointNet++..."
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

        model(
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
        f"{elapsed:.3f}s"
    )


del warmup_input


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
# FORCE ASYNC MODE
# ============================================================

settings = world.get_settings()


if settings.synchronous_mode:

    settings.synchronous_mode = False

    world.apply_settings(
        settings
    )

    print(
        "CARLA switched to asynchronous mode."
    )

else:

    print(
        "CARLA already asynchronous."
    )


# ============================================================
# SPAWN EGO TESLA
# ============================================================

vehicle_bp = (
    blueprints.find(
        "vehicle.tesla.model3"
    )
)


spawn_points = (
    world.get_map()
    .get_spawn_points()
)

random_spawn_point = random.choice(
    spawn_points
)

vehicle = world.try_spawn_actor(
    vehicle_bp,
    random_spawn_point
)

if vehicle is None:
    raise RuntimeError(
        "Could not spawn Tesla at random spawn point."
    )

print(
    "Tesla spawned at:",
    random_spawn_point.location
)
        


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
    3.0
)


traffic_manager.global_percentage_speed_difference(
    -10.0
)


# ============================================================
# BACKGROUND VEHICLES
# ============================================================

npc_vehicles = []


ego_location = (
    vehicle.get_location()
)


all_spawn_points = (
    world.get_map()
    .get_spawn_points()
)


safe_spawn_points = []


for spawn_point in (
    all_spawn_points
):

    distance = (
        spawn_point.location.distance(
            ego_location
        )
    )


    # Avoid spawning directly beside
    # the ego vehicle.

    if (
        30.0
        <= distance
        <= 120.0
    ):

        safe_spawn_points.append(
            spawn_point
        )


random.shuffle(
    safe_spawn_points
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


for spawn_point in (
    safe_spawn_points
):

    if (
        len(npc_vehicles)
        >=
        NUM_BACKGROUND_VEHICLES
    ):

        break


    npc_bp = random.choice(
        npc_blueprints
    )


    npc = (
        world.try_spawn_actor(
            npc_bp,
            spawn_point
        )
    )


    if npc is None:

        continue


    npc.set_autopilot(
        True,
        traffic_manager.get_port()
    )

    traffic_manager.ignore_lights_percentage(npc, 100)
    traffic_manager.ignore_signs_percentage(npc, 100)

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
            world
            .get_random_location_from_navigation()
        )


        if candidate is None:

            continue


        distance = (
            candidate.distance(
                ego_location
            )
        )


        if (
            8.0
            <= distance
            <= 30.0
        ):

            location = candidate

            break


    if location is None:

        continue


    walker_bp = random.choice(
        walker_blueprints
    )


    walker = (
        world.try_spawn_actor(
            walker_bp,
            carla.Transform(
                location
            )
        )
    )


    if walker is None:

        continue


    controller = (
        world.spawn_actor(
            walker_controller_bp,
            carla.Transform(),
            attach_to=walker
        )
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

for controller in (
    walker_controllers
):

    controller.start()


    # Try to give the pedestrian a destination that is still
# relatively close to the Tesla, so they remain visible
# during the demonstration.

destination = None

for _ in range(20):

    candidate_destination = (
        world.get_random_location_from_navigation()
    )

    if candidate_destination is None:
        continue

    destination_distance = candidate_destination.distance(
        vehicle.get_location()
    )

    if 10.0 <= destination_distance <= 35.0:
        destination = candidate_destination
        break

    if destination is not None:
        controller.go_to_location(destination)


    if destination is not None:

        controller.go_to_location(
            destination
        )


    controller.set_max_speed(
        random.uniform(
            1.0,
            1.5
        )
    )


# ============================================================
# TRAFFIC CHECK
# ============================================================

print()
print("=" * 55)
print("TRAFFIC SANITY CHECK")
print("=" * 55)

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
    "Total actors:",
    (
        1
        +
        len(npc_vehicles)
        +
        len(walkers)
    )
)


# ============================================================
# LiDAR
# ============================================================

lidar_bp = (
    blueprints.find(
        "sensor.lidar.ray_cast"
    )
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


lidar = (
    world.spawn_actor(
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
    f"{LIDAR_RANGE}m | "
    f"{LIDAR_ROTATION_FREQUENCY}Hz | "
    f"{LIDAR_SENSOR_TICK}s tick"
)


# ============================================================
# SHARED LIVE LiDAR DATA
#
# IMPORTANT:
# Store the CARLA sensor transform together with
# the LiDAR points.
# ============================================================

latest_xyz = None

latest_world_transform = None

latest_frame = None

frames_received = 0


data_lock = threading.Lock()


# ============================================================
# LiDAR CALLBACK
# ============================================================

def lidar_callback(data):

    global latest_xyz

    global latest_world_transform

    global latest_frame

    global frames_received


    raw = np.frombuffer(
        data.raw_data,
        dtype=np.float32
    )


    raw = raw.reshape(
        (-1, 4)
    )


    xyz = (
        raw[
            :,
            :3
        ]
        .astype(
            np.float32
        )
    )


    valid = np.isfinite(
        xyz
    ).all(
        axis=1
    )


    xyz = xyz[
        valid
    ]


    if len(xyz) == 0:

        return


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # data.transform is the LiDAR sensor's world transform
    # at the exact measurement time.
    # --------------------------------------------------------

    transform_matrix = np.asarray(
        data.transform.get_matrix(),
        dtype=np.float32
    )

    # Store a small world-coordinate sample for continuous map rendering.
    # This is separate from PointNet++ and is intentionally lightweight.
    world_xyz = transform_points_to_world(
        xyz,
        transform_matrix
    )

    if world_xyz is not None and len(world_xyz) > 0:
        # Avoid excessive plotting cost if a dense LiDAR frame arrives.
        if len(world_xyz) > 1200:
            keep = np.random.default_rng().choice(
                len(world_xyz),
                1200,
                replace=False
            )
            world_sample = world_xyz[keep]
        else:
            world_sample = world_xyz

        with data_history_lock:
            lidar_world_history.append(
                (time.time(), world_sample.astype(np.float32, copy=True))
            )


    with data_lock:

        latest_xyz = (
            xyz.copy()
        )

        latest_world_transform = (
            transform_matrix
        )

        latest_frame = (
            data.frame
        )

        frames_received += 1


# ============================================================
# LOCAL → WORLD TRANSFORM
# ============================================================

def transform_points_to_world(
    points,
    transform_matrix
):

    if (
        points is None
        or
        transform_matrix is None
    ):

        return None


    n = len(points)


    homogeneous = np.ones(
        (
            n,
            4
        ),
        dtype=np.float32
    )


    homogeneous[
        :,
        :3
    ] = points


    world_points = (
        homogeneous
        @
        transform_matrix.T
    )


    return (
        world_points[
            :,
            :3
        ]
        .astype(
            np.float32
        )
    )


# ============================================================
# POINTNET PREPROCESSING
# ============================================================

def preprocess_for_pointnet(
    xyz
):

    if (
        xyz is None
        or
        len(xyz) == 0
    ):

        return None


    n = len(xyz)


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


    sampled = (
        xyz[
            indices
        ]
        .astype(
            np.float32
        )
    )


    # --------------------------------------------------------
    # Match Jay's PointNet++ preprocessing.
    # --------------------------------------------------------

    centroid = np.mean(
        sampled,
        axis=0,
        keepdims=True
    )


    normalized = (
        sampled
        -
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
        sampled,
        normalized
    )


# ============================================================
# POINTNET INFERENCE
# ============================================================

def run_pointnet(
    xyz
):

    result = (
        preprocess_for_pointnet(
            xyz
        )
    )


    if result is None:

        return None


    sampled = result[0]

    normalized = result[1]


    tensor = (
        torch.from_numpy(
            normalized
        )
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )


    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    start = (
        time.perf_counter()
    )


    with torch.inference_mode():

        logits = model(
            tensor
        )


        predictions = (
            torch.argmax(
                logits,
                dim=-1
            )
        )


    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    elapsed = (
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
            sampled,

        "predictions":
            predictions,

        "inference_time":
            elapsed
    }


# ============================================================
# DYNAMIC OBJECT CLUSTERING
# ============================================================

def cluster_dynamic_objects(
    world_points,
    predictions
):

    dynamic_mask = (
        predictions == 3
    )


    dynamic_points = (
        world_points[
            dynamic_mask
        ]
    )


    if len(dynamic_points) < (
        DBSCAN_MIN_POINTS
    ):

        return []


    import open3d as o3d


    cloud = (
        o3d.geometry.PointCloud()
    )


    cloud.points = (
        o3d.utility.Vector3dVector(
            dynamic_points
        )
    )


    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=False
        )
    )


    objects = []


    if len(labels) == 0:

        return objects


    for cluster_id in sorted(
        set(labels)
    ):

        if cluster_id < 0:

            continue


        cluster = (
            dynamic_points[
                labels ==
                cluster_id
            ]
        )


        if len(cluster) < (
            DBSCAN_MIN_POINTS
        ):

            continue


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
            max_bound
            -
            min_bound
        )


        # Reject only microscopic noise.

        if (
            dimensions[0] < 0.15
            and
            dimensions[1] < 0.15
        ):

            continue


        center = (
            min_bound
            +
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
                    len(cluster)
            }
        )


    return objects


# ============================================================
# ADAPTIVE 2.5D CELL GENERATION
# ============================================================

def create_adaptive_cells(
    points,
    predictions
):

    if (
        points is None
        or
        len(points) == 0
    ):

        return []


    x = points[:, 0]

    y = points[:, 1]

    z = points[:, 2]


    # --------------------------------------------------------
    # Stable world-grid origin.
    # --------------------------------------------------------

    base_x_min = (
        np.floor(
            np.min(x)
            /
            BASE_CELL_SIZE
        )
        *
        BASE_CELL_SIZE
    )


    base_y_min = (
        np.floor(
            np.min(y)
            /
            BASE_CELL_SIZE
        )
        *
        BASE_CELL_SIZE
    )


    base_x_index = np.floor(
        (
            x
            -
            base_x_min
        )
        /
        BASE_CELL_SIZE
    ).astype(
        np.int32
    )


    base_y_index = np.floor(
        (
            y
            -
            base_y_min
        )
        /
        BASE_CELL_SIZE
    ).astype(
        np.int32
    )


    base_keys = np.column_stack(
        [
            base_x_index,
            base_y_index
        ]
    )


    unique_keys = np.unique(
        base_keys,
        axis=0
    )


    cells = []


    # ========================================================
    # PROCESS BASE CELLS
    # ========================================================

    for key in unique_keys:

        bx = int(
            key[0]
        )

        by = int(
            key[1]
        )


        mask = (
            (
                base_x_index
                ==
                bx
            )
            &
            (
                base_y_index
                ==
                by
            )
        )


        cell_points = (
            points[
                mask
            ]
        )


        cell_classes = (
            predictions[
                mask
            ]
        )


        if len(cell_points) == 0:

            continue


        class_counts = (
            np.bincount(
                cell_classes,
                minlength=NUM_CLASSES
            )
        )


        dominant_class = int(
            np.argmax(
                class_counts
            )
        )


        # ----------------------------------------------------
        # Adaptive resolution.
        # ----------------------------------------------------

        if dominant_class in {
            2,
            3,
            4
        }:

            target_size = (
                FINE_CELL_SIZE
            )

        elif dominant_class in {
            1,
            5
        }:

            target_size = (
                MEDIUM_CELL_SIZE
            )

        else:

            target_size = (
                BASE_CELL_SIZE
            )


        origin_x = (
            base_x_min
            +
            bx *
            BASE_CELL_SIZE
        )


        origin_y = (
            base_y_min
            +
            by *
            BASE_CELL_SIZE
        )


        # ====================================================
        # BASE-SIZE CELL
        # ====================================================

        if (
            target_size
            ==
            BASE_CELL_SIZE
        ):

            height = float(
                np.median(
                    cell_points[:, 2]
                )
            )


            cells.append(
                {
                    "x":
                        origin_x,

                    "y":
                        origin_y,

                    "size":
                        BASE_CELL_SIZE,

                    "class":
                        dominant_class,

                    "height":
                        height,

                    "points":
                        len(cell_points)
                }
            )


            continue


        # ====================================================
        # REFINED SUBCELLS
        # ====================================================

        sub_x = np.floor(
            (
                cell_points[:, 0]
                -
                origin_x
            )
            /
            target_size
        ).astype(
            np.int32
        )


        sub_y = np.floor(
            (
                cell_points[:, 1]
                -
                origin_y
            )
            /
            target_size
        ).astype(
            np.int32
        )


        sub_keys = np.column_stack(
            [
                sub_x,
                sub_y
            ]
        )


        unique_subkeys = np.unique(
            sub_keys,
            axis=0
        )


        for subkey in unique_subkeys:

            sx = int(
                subkey[0]
            )

            sy = int(
                subkey[1]
            )


            submask = (
                (
                    sub_x
                    ==
                    sx
                )
                &
                (
                    sub_y
                    ==
                    sy
                )
            )


            sub_points = (
                cell_points[
                    submask
                ]
            )


            sub_classes = (
                cell_classes[
                    submask
                ]
            )


            if len(sub_points) == 0:

                continue


            sub_counts = (
                np.bincount(
                    sub_classes,
                    minlength=NUM_CLASSES
                )
            )


            sub_class = int(
                np.argmax(
                    sub_counts
                )
            )


            height = float(
                np.median(
                    sub_points[:, 2]
                )
            )


            cells.append(
                {
                    "x":
                        origin_x
                        +
                        sx *
                        target_size,

                    "y":
                        origin_y
                        +
                        sy *
                        target_size,

                    "size":
                        target_size,

                    "class":
                        sub_class,

                    "height":
                        height,

                    "points":
                        len(sub_points)
                }
            )


    return cells


# ============================================================
# MOVING CARLA ACTOR BOXES FOR DEMO VISUALIZATION
# ============================================================

def get_dynamic_actor_boxes(
    ego_vehicle,
    npc_vehicles,
    walkers,
    latest_semantic_points=None,
    latest_semantic_predictions=None
):
    """
    Return world-aligned 2D boxes for clearly moving simulated actors.

    This is a visualization/ground-truth support layer for the CARLA
    prototype. PointNet++ remains responsible for the 7-class semantic
    predictions. The boxes are deliberately aligned with the moving
    actors visible in CARLA so the prototype video is visually coherent.
    """
    if ego_vehicle is None:
        return []

    ego_location = ego_vehicle.get_location()

    actors = []

    # Vehicles are the main dynamic objects we want to show clearly.
    for actor in npc_vehicles:
        actors.append((actor, "vehicle"))

    # Pedestrians are also dynamic, but we keep their boxes only when
    # they are actually moving and reasonably close to the ego vehicle.
    for actor in walkers:
        actors.append((actor, "pedestrian"))

    boxes = []

    for actor, kind in actors:
        try:
            if not actor.is_alive:
                continue

            transform = actor.get_transform()
            location = transform.location

            distance = float(
                np.hypot(
                    location.x - ego_location.x,
                    location.y - ego_location.y
                )
            )

            if distance > DYNAMIC_ACTOR_MAX_DISTANCE:
                continue

            velocity = actor.get_velocity()
            speed = float(
                np.sqrt(
                    velocity.x ** 2
                    + velocity.y ** 2
                    + velocity.z ** 2
                )
            )

            # For pedestrians use a slightly higher threshold to avoid
            # drawing boxes for barely moving/stationary walkers.
            min_speed = (
                DYNAMIC_ACTOR_MIN_SPEED
                if kind == "vehicle"
                else max(DYNAMIC_ACTOR_MIN_SPEED, 0.8)
            )

            if speed < min_speed:
                continue

            # Get the actor's world-space bounding-box vertices.
            vertices = actor.bounding_box.get_world_vertices(transform)

            xs = np.asarray(
                [v.x for v in vertices],
                dtype=np.float32
            )
            ys = np.asarray(
                [v.y for v in vertices],
                dtype=np.float32
            )
            zs = np.asarray(
                [v.z for v in vertices],
                dtype=np.float32
            )

            if len(xs) == 0:
                continue

            min_bound = np.array(
                [
                    float(xs.min()) - DYNAMIC_ACTOR_BOX_MARGIN,
                    float(ys.min()) - DYNAMIC_ACTOR_BOX_MARGIN,
                    float(zs.min())
                ],
                dtype=np.float32
            )

            max_bound = np.array(
                [
                    float(xs.max()) + DYNAMIC_ACTOR_BOX_MARGIN,
                    float(ys.max()) + DYNAMIC_ACTOR_BOX_MARGIN,
                    float(zs.max())
                ],
                dtype=np.float32
            )

            center = (min_bound + max_bound) / 2.0

            # Optional PointNet++ support count: count latest class-3
            # semantic points close to this actor's 2D bounding box.
            pn_support = 0

            if (
                latest_semantic_points is not None
                and latest_semantic_predictions is not None
                and
                len(latest_semantic_points)
                == len(latest_semantic_predictions)
            ):
                dynamic_mask = latest_semantic_predictions == 3

                if np.any(dynamic_mask):
                    dynamic_points = latest_semantic_points[
                        dynamic_mask
                    ]

                    support_radius = DYNAMIC_ACTOR_LIDAR_SUPPORT_RADIUS

                    support_mask = (
                        (dynamic_points[:, 0] >= min_bound[0] - support_radius)
                        &
                        (dynamic_points[:, 0] <= max_bound[0] + support_radius)
                        &
                        (dynamic_points[:, 1] >= min_bound[1] - support_radius)
                        &
                        (dynamic_points[:, 1] <= max_bound[1] + support_radius)
                    )

                    pn_support = int(np.count_nonzero(support_mask))

            boxes.append(
                {
                    "min": min_bound,
                    "max": max_bound,
                    "center": center,
                    "speed": speed,
                    "kind": kind,
                    "distance": distance,
                    "pn_dynamic_support": pn_support
                }
            )

        except Exception:
            # An individual actor disappearing during CARLA shutdown
            # should not break the entire visualization loop.
            continue

    # Closest actors first, then cap the number shown to keep the map clean.
    boxes.sort(key=lambda item: item["distance"])

    return boxes[:10]


# ============================================================
# MAP DRAWING
# ============================================================

def display_y(y):
    """
    Convert CARLA world Y into the visualization coordinate.

    This is ONLY a display transformation.
    The underlying CARLA world coordinates remain unchanged.
    """

    if DISPLAY_MIRROR_Y:
        return -y

    return y


# ============================================================
# BUILD CLEAN SEMANTIC 2.5D CELLS
# ============================================================

def build_semantic_cells(
    records,
    current_time
):

    if not records:
        return []


    cell_data = {}


    for record in records:

        age = (
            current_time -
            record["time"]
        )


        predictions = record[
            "predictions"
        ]

        points = record[
            "world_points"
        ]


        # ----------------------------------------------------
        # Dynamic objects use only short history.
        # ----------------------------------------------------

        if age > DYNAMIC_DISPLAY_SECONDS:

            mask = (
                predictions != 3
            )

        else:

            mask = np.ones(
                len(predictions),
                dtype=bool
            )


        points = points[
            mask
        ]

        predictions = predictions[
            mask
        ]


        for point, class_id in zip(
            points,
            predictions
        ):

            class_id = int(
                class_id
            )


            # ------------------------------------------------
            # Adaptive resolution.
            # ------------------------------------------------

            if class_id in {
                2,
                3,
                4
            }:

                cell_size = (
                    FINE_CELL_SIZE
                )

            elif class_id in {
                1,
                5
            }:

                cell_size = (
                    MEDIUM_CELL_SIZE
                )

            else:

                cell_size = (
                    BASE_CELL_SIZE
                )


            gx = int(
                np.floor(
                    point[0] /
                    cell_size
                )
            )


            gy = int(
                np.floor(
                    point[1] /
                    cell_size
                )
            )


            key = (
                gx,
                gy,
                cell_size
            )


            if key not in cell_data:

                cell_data[key] = {
                    "classes": [],
                    "heights": []
                }


            cell_data[key]["classes"].append(
                class_id
            )

            cell_data[key]["heights"].append(
                float(point[2])
            )


    cells = []


    # ========================================================
    # CONVERT CELL STATISTICS
    # ========================================================

    for (
        key,
        data
    ) in cell_data.items():

        gx, gy, cell_size = key


        if len(
            data["classes"]
        ) < MIN_CELL_POINTS:

            continue


        classes = np.asarray(
            data["classes"],
            dtype=np.int32
        )


        heights = np.asarray(
            data["heights"],
            dtype=np.float32
        )


        class_counts = np.bincount(
            classes,
            minlength=NUM_CLASSES
        )


        dominant_class = int(
            np.argmax(
                class_counts
            )
        )


        # ----------------------------------------------------
        # Stable world-grid cell origin.
        # ----------------------------------------------------

        cell_x = (
            gx *
            cell_size
        )


        cell_y = (
            gy *
            cell_size
        )


        # Median is much less sensitive to isolated
        # high LiDAR returns than max height.

        median_height = float(
            np.median(
                heights
            )
        )


        cells.append(
            {
                "x":
                    cell_x,

                "y":
                    cell_y,

                "size":
                    cell_size,

                "class":
                    dominant_class,

                "height":
                    median_height,

                "points":
                    len(classes)
            }
        )


    return cells


# ============================================================
# CLEAN SEMANTIC MAP DRAWING
# ============================================================

def draw_semantic_map(
    ax,
    cells,
    dynamic_objects,
    dynamic_actor_boxes,
    ego_position,
    ego_yaw,
    trajectory,
    current_frame,
    elapsed_seconds,
    inference_time,
    background_points=None,
    latest_semantic_points=None,
    latest_semantic_predictions=None
):

    ax.clear()

    # ========================================================
    # CONTINUOUS LiDAR BACKGROUND
    # ========================================================
    # This is raw world-coordinate LiDAR only. It provides a stable
    # spatial skeleton even before the next PointNet++ semantic update.
    if background_points is not None and len(background_points) > 0:
        display_background_y = (
            -background_points[:, 1]
            if DISPLAY_MIRROR_Y
            else background_points[:, 1]
        )

        ax.scatter(
            background_points[:, 0],
            display_background_y,
            s=2.0,
            c="#B0B0B0",
            alpha=0.22,
            linewidths=0,
            zorder=1,
            label="Live LiDAR"
        )

    # ========================================================
    # DRAW SEMANTIC CELLS
    # ========================================================
    for cell in cells:

        class_id = int(cell["class"])
        x = float(cell["x"])
        y = float(cell["y"])
        size = float(cell["size"])

        if DISPLAY_MIRROR_Y:
            plot_y = -y - size
        else:
            plot_y = y

        rect = Rectangle(
            (x, plot_y),
            size,
            size,
            facecolor=CLASS_COLORS[class_id],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.78,
            zorder=3
        )

        ax.add_patch(rect)

    # ========================================================
    # SEMANTIC POINT FALLBACK
    # ========================================================
    # In case adaptive cells are temporarily sparse, show the newest
    # classified sample directly. This prevents a blank semantic panel.
    if (
        len(cells) == 0
        and
        latest_semantic_points is not None
        and
        latest_semantic_predictions is not None
        and
        len(latest_semantic_points) == len(latest_semantic_predictions)
    ):
        for class_id in range(NUM_CLASSES):
            mask = latest_semantic_predictions == class_id
            if not np.any(mask):
                continue

            plot_y = (
                -latest_semantic_points[mask, 1]
                if DISPLAY_MIRROR_Y
                else latest_semantic_points[mask, 1]
            )

            ax.scatter(
                latest_semantic_points[mask, 0],
                plot_y,
                s=8.0,
                c=CLASS_COLORS[class_id],
                alpha=0.80,
                linewidths=0,
                zorder=4
            )

    # ========================================================
    # POINTNET++ DYNAMIC SEMANTIC EVIDENCE
    # ========================================================
    # Class 3 is the model's "Dynamic object" class.
    # Show the newest class-3 points explicitly in orange so the
    # semantic prediction itself is always visible.
    if (
        latest_semantic_points is not None
        and
        latest_semantic_predictions is not None
        and
        len(latest_semantic_points) == len(latest_semantic_predictions)
    ):
        dynamic_mask = latest_semantic_predictions == 3

        if np.any(dynamic_mask):
            dynamic_plot_y = (
                -latest_semantic_points[dynamic_mask, 1]
                if DISPLAY_MIRROR_Y
                else latest_semantic_points[dynamic_mask, 1]
            )

            ax.scatter(
                latest_semantic_points[dynamic_mask, 0],
                dynamic_plot_y,
                s=12.0,
                c=CLASS_COLORS[3],
                alpha=0.75,
                linewidths=0,
                zorder=5,
                label="PointNet++ dynamic"
            )

    # ========================================================
    # MOVING-ACTOR DYNAMIC BOXES
    # ========================================================
    # These boxes are aligned to the actual moving CARLA actors so
    # the visualization matches the vehicle/fire-truck motion seen
    # in the simulation. They are not claimed to be PointNet++ boxes.
    for i, obj in enumerate(dynamic_actor_boxes):

        min_bound = np.asarray(obj["min"], dtype=np.float32)
        max_bound = np.asarray(obj["max"], dtype=np.float32)

        x1 = float(min_bound[0])
        x2 = float(max_bound[0])
        y1 = float(min_bound[1])
        y2 = float(max_bound[1])

        if DISPLAY_MIRROR_Y:
            plot_y = -y2
        else:
            plot_y = y1

        box_width = max(x2 - x1, 0.8)
        box_height = max(y2 - y1, 0.8)

        # Orange outline + light transparent fill.
        rect = Rectangle(
            (x1, plot_y),
            box_width,
            box_height,
            fill=True,
            facecolor=CLASS_COLORS[3],
            edgecolor=CLASS_COLORS[3],
            linewidth=2.5,
            alpha=0.12,
            zorder=9
        )

        ax.add_patch(rect)

        # Strong outline on top.
        outline = Rectangle(
            (x1, plot_y),
            box_width,
            box_height,
            fill=False,
            edgecolor=CLASS_COLORS[3],
            linewidth=2.5,
            zorder=10
        )

        ax.add_patch(outline)

        center = obj["center"]
        label_y = (
            -float(center[1])
            if DISPLAY_MIRROR_Y
            else float(center[1])
        )

        speed_kmh = float(obj.get("speed", 0.0)) * 3.6
        actor_kind = obj.get("kind", "vehicle")

        support = int(obj.get("pn_dynamic_support", 0))

        if support > 0:
            label = f"D{i + 1}  {speed_kmh:.0f} km/h  PN++:{support}"
        else:
            label = f"D{i + 1}  {speed_kmh:.0f} km/h"

        ax.text(
            float(center[0]),
            label_y + box_height / 2.0 + 0.9,
            label,
            fontsize=7.5,
            fontweight="bold",
            color=CLASS_COLORS[3],
            ha="center",
            zorder=11,
            bbox={
                "facecolor": "white",
                "edgecolor": CLASS_COLORS[3],
                "alpha": 0.90,
                "pad": 1.5
            }
        )

    # ========================================================
    # OPTIONAL: POINTNET++ CLUSTERS ARE NOT DRAWN AS VEHICLE BOXES
    # ========================================================
    # dynamic_objects still exists as diagnostic model output.
    # We intentionally do not draw these DBSCAN clusters as if they
    # were guaranteed to be vehicles, because this model is semantic
    # segmentation, not a vehicle detector.

    # ========================================================
    # TESLA TRAJECTORY
    # ========================================================
    if len(trajectory) >= 2:
        trajectory_array = np.asarray(trajectory)
        trajectory_x = trajectory_array[:, 0]
        trajectory_y = trajectory_array[:, 1]

        if DISPLAY_MIRROR_Y:
            trajectory_y = -trajectory_y

        ax.plot(
            trajectory_x,
            trajectory_y,
            linewidth=2.5,
            color="#1565C0",
            label="Tesla trajectory",
            zorder=15
        )

    # ========================================================
    # EGO TESLA POSITION + HEADING
    # ========================================================
    if ego_position is not None:
        ego_x = float(ego_position[0])
        ego_y = float(ego_position[1])

        if DISPLAY_MIRROR_Y:
            ego_y_plot = -ego_y
        else:
            ego_y_plot = ego_y

        ax.scatter(
            [ego_x],
            [ego_y_plot],
            s=110,
            marker="o",
            c="#1565C0",
            edgecolors="black",
            linewidths=1.2,
            zorder=20
        )

        arrow_length = 6.0
        yaw_rad = np.deg2rad(ego_yaw)
        dx = np.cos(yaw_rad) * arrow_length
        dy = np.sin(yaw_rad) * arrow_length

        if DISPLAY_MIRROR_Y:
            dy = -dy

        ax.arrow(
            ego_x,
            ego_y_plot,
            dx,
            dy,
            width=0.15,
            head_width=1.0,
            head_length=1.5,
            length_includes_head=True,
            color="#1565C0",
            zorder=21
        )

    # ========================================================
    # LEGEND
    # ========================================================
    handles = []

    for class_id in range(NUM_CLASSES):
        handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=CLASS_COLORS[class_id],
                edgecolor="black",
                label=f"{class_id} - {CLASS_NAMES[class_id]}"
            )
        )

    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.92
    )

    # ========================================================
    # TITLE
    # ========================================================
    ax.set_title(
        "LIVE SEMANTIC ADAPTIVE 2.5D LiDAR MAP\n"
        f"CARLA {elapsed_seconds:04.1f}s | "
        f"Frame {current_frame} | "
        f"PointNet++ {inference_time:.2f}s",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.set_facecolor("#F5F5F5")

    # ========================================================
    # STATUS BOX
    # ========================================================
    dynamic_count = len(dynamic_actor_boxes)
    pointnet_dynamic_clusters = len(dynamic_objects)
    dynamic_points = 0
    vegetation_points = 0
    static_points = 0

    class_evidence = [0] * NUM_CLASSES
    for cell in cells:
        class_evidence[int(cell["class"])] += int(cell["points"])

    dynamic_points = class_evidence[3]
    vegetation_points = class_evidence[4]
    static_points = class_evidence[2]

    ax.text(
        0.015,
        0.015,
        (
            f"Dynamic actors: {dynamic_count}\n"
            f"PN++ dynamic evidence: {dynamic_points}\n"
            f"PN++ clusters: {pointnet_dynamic_clusters}\n"
            f"Vegetation evidence: {vegetation_points}\n"
            f"Static obstacle evidence: {static_points}\n"
            f"Semantic cells: {len(cells)}\n\n"
            "Orange = Dynamic object (class 3)\n"
            "Boxes = moving CARLA actors + LiDAR support\n"
            "Adaptive resolution: 2.0 m  →  1.0 m  →  0.5 m"
        ),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "#555555",
            "alpha": 0.90
        },
        zorder=30
    )

    ax.text(
        0.99,
        0.015,
        "WORLD-ALIGNED\nTOP-DOWN VIEW",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="#444444",
        bbox={
            "facecolor": "white",
            "alpha": 0.75,
            "pad": 2
        }
    )


# ============================================================
# START SHARED DATA
# ============================================================

# Semantic predictions are produced only every POINTNET_INTERVAL seconds.
# Keep a bounded history so the 2.5D map can accumulate the environment
# without growing forever.
semantic_history = deque(maxlen=300)

# A lightweight world-coordinate LiDAR history is also kept as a visual
# fallback/background. This guarantees that the map is never visually empty
# while PointNet++ is warming up or between semantic updates.
lidar_world_history = deque(maxlen=160)

latest_semantic_result = None
latest_prediction_frame = None

last_prediction_wall_time = (
    0.0
)

prediction_lock = (
    threading.Lock()
)

data_history_lock = (
    threading.Lock()
)


stop_event = (
    threading.Event()
)


# ============================================================
# POINTNET WORKER
#
# It performs inference periodically.
# ============================================================

def pointnet_worker():

    global latest_semantic_result
    global last_prediction_wall_time
    global latest_prediction_frame
    global latest_objects


    print()

    print(
        "PointNet++ worker started."
    )


    last_processed_frame = None


    while not stop_event.is_set():

        time.sleep(
            0.02
        )


        # ----------------------------------------------------
        # Wait until next prediction slot.
        # ----------------------------------------------------

        if (
            time.time()
            -
            last_prediction_wall_time
            <
            POINTNET_INTERVAL
        ):

            continue


        # ----------------------------------------------------
        # Get newest LiDAR packet.
        # ----------------------------------------------------

        with data_lock:

            if (
                latest_xyz is None
                or
                latest_world_transform
                is None
            ):

                continue


            local_xyz = (
                latest_xyz.copy()
            )


            transform_matrix = (
                latest_world_transform.copy()
            )


            frame_id = (
                latest_frame
            )


        if (
            frame_id is None
            or
            frame_id ==
            last_processed_frame
        ):

            continue


        last_processed_frame = (
            frame_id
        )


        # ----------------------------------------------------
        # Transform ALL raw points into world coordinates.
        #
        # This is the critical fix.
        # ----------------------------------------------------

        world_xyz = (
            transform_points_to_world(
                local_xyz,
                transform_matrix
            )
        )


        if (
            world_xyz is None
            or
            len(world_xyz) == 0
        ):

            continue


        # ----------------------------------------------------
        # PointNet++
        # ----------------------------------------------------

        result = run_pointnet(
            local_xyz
        )


        if result is None:

            continue


        local_sampled = result[
            "points"
        ]


        predictions = result[
            "predictions"
        ]


        inference_time = result[
            "inference_time"
        ]


        # ----------------------------------------------------
        # Transform ONLY the sampled points that were
        # actually classified by PointNet++.
        # ----------------------------------------------------

        sampled_world = (
            transform_points_to_world(
                local_sampled,
                transform_matrix
            )
        )


        if sampled_world is None:

            continue


        # ----------------------------------------------------
        # Dynamic objects.
        # ----------------------------------------------------

        objects = (
            cluster_dynamic_objects(
                sampled_world,
                predictions
            )
        )


        # ----------------------------------------------------
        # Store semantic result.
        # ----------------------------------------------------

        result_record = {

            "time":
                time.time(),

            "frame":
                frame_id,

            "world_points":
                sampled_world,

            "predictions":
                predictions,

            "objects":
                objects,

            "inference_time":
                inference_time
        }


        with prediction_lock:

            latest_semantic_result = (
                result_record
            )

            semantic_history.append(
                result_record
            )


        last_prediction_wall_time = (
            time.time()
        )


        # ----------------------------------------------------
        # Print result.
        # ----------------------------------------------------

        counts = np.bincount(

            predictions,

            minlength=NUM_CLASSES
        )


        print()
        print(
            "--------------------------------------------"
        )

        print(
            "POINTNET++ SEMANTIC UPDATE"
        )

        print(
            "Frame:",
            frame_id
        )

        print(
            "Inference:",
            f"{inference_time:.3f}s"
        )

        print(
            "Dynamic:",
            int(counts[3])
        )

        print(
            "Vegetation:",
            int(counts[4])
        )

        print(
            "Static obstacle:",
            int(counts[2])
        )

        print(
            "Dynamic clusters:",
            len(objects)
        )


# ============================================================
# START LiDAR
# ============================================================

lidar.listen(
    lidar_callback
)


# ============================================================
# START POINTNET WORKER
# ============================================================

worker_thread = (
    threading.Thread(
        target=pointnet_worker,
        daemon=True
    )
)


worker_thread.start()


# ============================================================
# START EGO AUTOPILOT
# ============================================================

vehicle.set_autopilot(
    True,
    traffic_manager.get_port()
)

traffic_manager.ignore_lights_percentage(vehicle, 100)
traffic_manager.ignore_signs_percentage(vehicle, 100)

# ============================================================
# MATPLOTLIB
# ============================================================

plt.ion()


fig, ax = plt.subplots(
    figsize=(12, 9)
)


# ============================================================
# CARLA SPECTATOR
# ============================================================

spectator = (
    world.get_spectator()
)


# ============================================================
# TRAJECTORY
# ============================================================

trajectory = deque(
    maxlen=3000
)


# ============================================================
# START DEMO
# ============================================================

print()
print("=" * 78)
print("SIH FINAL DEMO STARTED")
print("=" * 78)

print(
    "Tesla driving: YES"
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
    "LiDAR: WORLD-COORDINATE MAPPING"
)

print(
    "PointNet++: 7-CLASS SEMANTIC PREDICTION"
)

print(
    "Adaptive 2.5D: 2m / 1m / 0.5m"
)

print(
    "Semantic history:",
    f"{STATIC_HISTORY_SECONDS:.0f}s static / "
    f"{DYNAMIC_HISTORY_SECONDS:.0f}s dynamic"
)

print()

print(
    "Use NVIDIA ShadowPlay / Alt+Z to record."
)

print()


# ============================================================
# DEMO LOOP
# ============================================================

start_time = (
    time.time()
)


last_display = 0.0


last_frame_displayed = None


try:

    while (
        time.time()
        -
        start_time
        <
        COLLECTION_TIME
    ):

        # ====================================================
        # CAMERA FOLLOW
        # ====================================================

        vehicle_transform = (
            vehicle.get_transform()
        )


        vehicle_location = (
            vehicle_transform.location
        )


        vehicle_yaw = (
            vehicle_transform.rotation.yaw
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

                yaw=(
                    vehicle_transform
                    .rotation
                    .yaw
                ),

                roll=0.0
            )
        )


        spectator.set_transform(
            carla.Transform(
                camera_location,
                camera_rotation
            )
        )


        # ====================================================
        # EGO TRAJECTORY
        # ====================================================

        trajectory.append(
            (
                vehicle_location.x,
                vehicle_location.y
            )
        )


        # ====================================================
        # SNAPSHOT SHARED MAP DATA
        # ====================================================

        now = time.time()

        # Semantic history is shared with the PointNet worker.
        with prediction_lock:
            while (
                semantic_history
                and
                now - semantic_history[0]["time"] > STATIC_HISTORY_SECONDS
            ):
                semantic_history.popleft()

            semantic_records = list(semantic_history)
            latest_result = latest_semantic_result

        semantic_cells = build_semantic_cells(
            semantic_records,
            now
        )

        if latest_result is not None:
            latest_objects = latest_result["objects"]
            latest_inference = latest_result["inference_time"]
            latest_prediction_frame = latest_result["frame"]
            latest_semantic_points = latest_result["world_points"]
            latest_semantic_predictions = latest_result["predictions"]
        else:
            latest_objects = []
            latest_inference = 0.0
            latest_prediction_frame = None
            latest_semantic_points = None
            latest_semantic_predictions = None

        # ====================================================
        # MOVING ACTOR DYNAMIC BOXES
        # ====================================================
        # This makes obvious moving vehicles/fire trucks line up with
        # the orange dynamic visualization in the CARLA view.
        dynamic_actor_boxes = get_dynamic_actor_boxes(
            vehicle,
            npc_vehicles,
            walkers,
            latest_semantic_points,
            latest_semantic_predictions
        )

        # Snapshot recent raw world-coordinate LiDAR for a continuous
        # background map and remove stale entries.
        with data_history_lock:
            while (
                lidar_world_history
                and
                now - lidar_world_history[0][0] > STATIC_HISTORY_SECONDS
            ):
                lidar_world_history.popleft()

            lidar_records = list(lidar_world_history)

        if lidar_records:
            background_points = np.vstack(
                [points for _, points in lidar_records]
            )

            # Keep plotting light even if CARLA starts returning dense scans.
            if len(background_points) > 16000:
                keep = np.linspace(
                    0,
                    len(background_points) - 1,
                    16000,
                    dtype=np.int64
                )
                background_points = background_points[keep]
        else:
            background_points = None

        # ====================================================
        # UPDATE DISPLAY
        # ====================================================

        if (
            time.time()
            -
            last_display
            >=
            0.10
        ):

            elapsed = (
                time.time()
                -
                start_time
            )


            draw_semantic_map(

                ax,

                semantic_cells,

                latest_objects,

                dynamic_actor_boxes,

                np.array(
                    [
                        vehicle_location.x,
                        vehicle_location.y
                    ]
                ),

                vehicle_yaw,

                list(trajectory),

                latest_prediction_frame,

                elapsed,

                latest_inference,

                background_points,

                latest_semantic_points,

                latest_semantic_predictions
            )


            # ------------------------------------------------
            # Stable world-map window.
            #
            # Keep the world northing/orientation stable.
            # ------------------------------------------------

            # ------------------------------------------------------------
            # Choose a stable world-coordinate viewing window that contains
            # the ego vehicle, trajectory, and current LiDAR/semantic evidence.
            # The axes never rotate with the Tesla.
            # ------------------------------------------------------------
            visible_x = [float(vehicle_location.x)]
            visible_y = [
                float(
                    -vehicle_location.y
                    if DISPLAY_MIRROR_Y
                    else vehicle_location.y
                )
            ]

            if len(trajectory) >= 1:
                trajectory_array = np.asarray(trajectory)
                visible_x.extend(trajectory_array[:, 0].tolist())

                trajectory_y = trajectory_array[:, 1]
                if DISPLAY_MIRROR_Y:
                    trajectory_y = -trajectory_y
                visible_y.extend(trajectory_y.tolist())

            if background_points is not None and len(background_points) > 0:
                visible_x.extend(background_points[:, 0].tolist())
                bg_y = (
                    -background_points[:, 1]
                    if DISPLAY_MIRROR_Y
                    else background_points[:, 1]
                )
                visible_y.extend(bg_y.tolist())

            if latest_semantic_points is not None and len(latest_semantic_points) > 0:
                visible_x.extend(latest_semantic_points[:, 0].tolist())
                sem_y = (
                    -latest_semantic_points[:, 1]
                    if DISPLAY_MIRROR_Y
                    else latest_semantic_points[:, 1]
                )
                visible_y.extend(sem_y.tolist())

            x_low = min(visible_x) - 15.0
            x_high = max(visible_x) + 15.0
            y_low = min(visible_y) - 15.0
            y_high = max(visible_y) + 15.0

            # Prevent an overly tight window when the demo has only just started.
            if x_high - x_low < 80.0:
                center_x = (x_low + x_high) / 2.0
                x_low = center_x - 40.0
                x_high = center_x + 40.0

            if y_high - y_low < 80.0:
                center_y = (y_low + y_high) / 2.0
                y_low = center_y - 40.0
                y_high = center_y + 40.0

            ax.set_xlim(x_low, x_high)
            ax.set_ylim(y_low, y_high)


            fig.canvas.draw_idle()

            fig.canvas.flush_events()


            last_display = (
                time.time()
            )


        time.sleep(
            0.01
        )


finally:

    print()
    print(
        "Stopping SIH demo..."
    )


    stop_event.set()

    try:
        worker_thread.join(timeout=1.0)
    except Exception:
        pass


    try:

        lidar.stop()

    except Exception:

        pass


# ============================================================
# FINAL STATISTICS
# ============================================================

print()
print("=" * 78)
print("SIH FINAL DEMO COMPLETE")
print("=" * 78)

print(
    "LiDAR frames received:",
    frames_received
)


with prediction_lock:

    final_result = (
        latest_semantic_result
    )


if final_result is not None:

    print(
        "Last PointNet frame:",
        final_result["frame"]
    )

    print(
        "Last inference:",
        f"{final_result['inference_time']:.3f}s"
    )

    print(
        "Final dynamic clusters:",
        len(
            final_result[
                "objects"
            ]
        )
    )

else:

    print(
        "No completed PointNet prediction."
    )


# ============================================================
# SAVE FINAL REPORT GRAPH
# ============================================================

print()
print("=" * 78)
print("GENERATING FINAL 2.5D REPORT GRAPH")
print("=" * 78)

final_now = time.time()

# ------------------------------------------------------------
# Snapshot final semantic history
# ------------------------------------------------------------
with prediction_lock:
    final_semantic_records = list(semantic_history)
    final_result = latest_semantic_result

final_semantic_cells = build_semantic_cells(
    final_semantic_records,
    final_now
)

# ------------------------------------------------------------
# Snapshot final LiDAR history
# ------------------------------------------------------------
with data_history_lock:
    final_lidar_records = list(lidar_world_history)

if final_lidar_records:
    final_background_points = np.vstack(
        [points for _, points in final_lidar_records]
    )
else:
    final_background_points = None

# ------------------------------------------------------------
# Final PointNet++ result
# ------------------------------------------------------------
if final_result is not None:

    final_dynamic_objects = final_result.get(
        "objects",
        []
    )

    final_inference_time = float(
        final_result.get(
            "inference_time",
            0.0
        )
    )

    final_frame = final_result.get(
        "frame",
        None
    )

    final_semantic_points = final_result.get(
        "world_points",
        None
    )

    final_semantic_predictions = final_result.get(
        "predictions",
        None
    )

else:

    final_dynamic_objects = []

    final_inference_time = 0.0

    final_frame = None

    final_semantic_points = None

    final_semantic_predictions = None


# ------------------------------------------------------------
# Final moving CARLA actor boxes
# ------------------------------------------------------------
final_dynamic_actor_boxes = get_dynamic_actor_boxes(
    vehicle,
    npc_vehicles,
    walkers,
    final_semantic_points,
    final_semantic_predictions
)

# ------------------------------------------------------------
# Create a dedicated report figure
# ------------------------------------------------------------
report_fig, report_ax = plt.subplots(
    figsize=(12, 10)
)

# Draw using the SAME working map renderer
draw_semantic_map(
    report_ax,
    final_semantic_cells,
    final_dynamic_objects,
    final_dynamic_actor_boxes,
    np.array([
        vehicle.get_location().x,
        vehicle.get_location().y
    ]),
    vehicle.get_transform().rotation.yaw,
    list(trajectory),
    final_frame,
    COLLECTION_TIME,
    final_inference_time,
    final_background_points,
    final_semantic_points,
    final_semantic_predictions
)

# ------------------------------------------------------------
# Fit the final map around the complete trajectory + LiDAR
# ------------------------------------------------------------
visible_x = []
visible_y = []

if len(trajectory) > 0:

    trajectory_array = np.asarray(trajectory)

    visible_x.extend(
        trajectory_array[:, 0].tolist()
    )

    trajectory_y = trajectory_array[:, 1]

    if DISPLAY_MIRROR_Y:
        trajectory_y = -trajectory_y

    visible_y.extend(
        trajectory_y.tolist()
    )

if final_background_points is not None:

    visible_x.extend(
        final_background_points[:, 0].tolist()
    )

    background_y = (
        -final_background_points[:, 1]
        if DISPLAY_MIRROR_Y
        else final_background_points[:, 1]
    )

    visible_y.extend(
        background_y.tolist()
    )

if final_semantic_points is not None:

    visible_x.extend(
        final_semantic_points[:, 0].tolist()
    )

    semantic_y = (
        -final_semantic_points[:, 1]
        if DISPLAY_MIRROR_Y
        else final_semantic_points[:, 1]
    )

    visible_y.extend(
        semantic_y.tolist()
    )

if visible_x and visible_y:

    report_ax.set_xlim(
        min(visible_x) - 10,
        max(visible_x) + 10
    )

    report_ax.set_ylim(
        min(visible_y) - 10,
        max(visible_y) + 10
    )

# ------------------------------------------------------------
# Final report title
# ------------------------------------------------------------
report_ax.set_title(
    "FINAL ADAPTIVE 2.5D LiDAR SEMANTIC MAP\n"
    "CARLA Town10HD | 30s Drive | PointNet++ 7-Class Semantic Segmentation",
    fontsize=15,
    fontweight="bold",
    pad=15
)

# ------------------------------------------------------------
# Save high-resolution PNG
# ------------------------------------------------------------
report_path = "final_adaptive_2_5d_report.png"

report_fig.savefig(
    report_path,
    dpi=300,
    bbox_inches="tight"
)

print()
print("FINAL REPORT GRAPH SAVED:")
print(os.path.abspath(report_path))

plt.show(block=False)

# ============================================================
# CLEANUP LiDAR
# ============================================================

try:

    lidar.destroy()

except Exception:

    pass


# ============================================================
# CLEANUP PEDESTRIANS
# ============================================================

for controller in (
    walker_controllers
):

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


# ============================================================
# CLEANUP NPC VEHICLES
# ============================================================

for npc in npc_vehicles:

    try:

        npc.destroy()

    except Exception:

        pass


# ============================================================
# CLEANUP EGO
# ============================================================

try:

    vehicle.destroy()

except Exception:

    pass


plt.ioff()

plt.close(
    fig
)


print()
print(
    "All CARLA actors cleaned up."
)

print(
    "Experiment complete."
)
