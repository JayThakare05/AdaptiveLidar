import os
import random
import time
import csv

import numpy as np
import torch
import open3d as o3d

from train_pointnet2 import PointNet2Segmentation


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "pointnet2_best.pth"

NUM_CLASSES = 7
NUM_POINTS = 2048

COLLECTION_TIME = 20.0

NUM_BACKGROUND_VEHICLES = 15
NUM_PEDESTRIANS = 20

LIDAR_RANGE = 50.0

# DBSCAN parameters for predicted object points
DBSCAN_EPS = 1.5
DBSCAN_MIN_POINTS = 5


# ============================================================
# CLASS NAMES
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
# VISUALIZATION COLORS
# ============================================================

CLASS_COLORS = {
    0: [0.35, 0.35, 0.35],  # Drivable
    1: [0.65, 0.65, 0.65],  # Non-drivable
    2: [1.00, 0.10, 0.10],  # Static obstacle
    3: [1.00, 0.50, 0.00],  # Dynamic object
    4: [0.10, 0.70, 0.10],  # Vegetation
    5: [1.00, 0.85, 0.00],  # Road marking
    6: [0.60, 0.10, 0.70],  # Other
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
print("CARLA + POINTNET++ LIVE OBJECT DETECTION")
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
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{os.path.abspath(MODEL_PATH)}"
    )


# ============================================================
# LOAD POINTNET++ MODEL
# ============================================================

print()
print("Creating PointNet++ model...")

model = PointNet2Segmentation()

model = model.to(DEVICE)


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
# CONNECT TO CARLA
# ============================================================

import carla


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
        "Switched CARLA to asynchronous mode."
    )

else:

    print(
        "CARLA already asynchronous."
    )


# ============================================================
# SPAWN EGO TESLA
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

for spawn_point in spawn_points:

    vehicle = world.try_spawn_actor(
        vehicle_bp,
        spawn_point
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
    5.0
)

traffic_manager.global_percentage_speed_difference(
    15.0
)


# ============================================================
# CONTROLLED BACKGROUND VEHICLES
# ============================================================

npc_vehicles = []

print()
print(
    "Spawning background vehicles..."
)


vehicle_spawn_points = (
    world.get_map().get_spawn_points()
)

ego_location = (
    vehicle.get_location()
)


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


    if distance < 12.0:
        continue


    if distance > 70.0:
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

controller_bp = blueprints.find(
    "controller.ai.walker"
)


print(
    "Spawning pedestrians..."
)


for _ in range(
    NUM_PEDESTRIANS
):

    spawn_location = None


    for _ in range(30):

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
            8.0
            <= distance
            <= 60.0
        ):

            spawn_location = candidate

            break


    if spawn_location is None:
        continue


    walker_bp = random.choice(
        walker_blueprints
    )


    walker = world.try_spawn_actor(
        walker_bp,
        carla.Transform(
            spawn_location
        )
    )


    if walker is None:
        continue


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
# TRAFFIC SANITY CHECK
# ============================================================

print()
print(
    "============================================"
)

print(
    "TRAFFIC SANITY CHECK"
)

print(
    "============================================"
)

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

lidar_bp.set_attribute(
    "sensor_tick",
    "0.1"
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


# ============================================================
# STORAGE
# ============================================================

latest_points = None
latest_predictions = None
latest_boxes = []

best_points = None
best_predictions = None
best_boxes = []

best_dynamic_objects = 0

frame_counter = 0


# ============================================================
# PREDICT ONE LiDAR FRAME
# ============================================================

def predict_frame(
    xyz
):

    global model


    original_count = len(xyz)


    if original_count == 0:

        return (
            np.empty(
                (0, 3),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.int64
            )
        )


    # --------------------------------------------------------
    # MODEL REQUIRES 2048 POINTS
    # --------------------------------------------------------

    rng = np.random.default_rng()


    if original_count >= NUM_POINTS:

        sample_indices = (
            rng.choice(
                original_count,
                NUM_POINTS,
                replace=False
            )
        )

    else:

        # Sample with replacement
        # to match Jay's training logic
        sample_indices = (
            rng.choice(
                original_count,
                NUM_POINTS,
                replace=True
            )
        )


    sampled_xyz = (
        xyz[
            sample_indices
        ].astype(
            np.float32
        )
    )


    # --------------------------------------------------------
    # NORMALIZATION
    # Same as training
    # --------------------------------------------------------

    centroid = np.mean(
        sampled_xyz,
        axis=0,
        keepdims=True
    )

    normalized = (
        sampled_xyz -
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


    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    input_tensor = (
        torch.from_numpy(
            normalized
        )
        .unsqueeze(0)
        .to(DEVICE)
    )


    with torch.no_grad():

        logits = model(
            input_tensor
        )

        sample_predictions = (
            torch.argmax(
                logits,
                dim=-1
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )


    # --------------------------------------------------------
    # MAP SAMPLE PREDICTIONS BACK TO
    # THE ORIGINAL LiDAR POINTS
    # --------------------------------------------------------

    votes = [
        {}
        for _ in range(
            original_count
        )
    ]


    for point_idx, prediction in zip(
        sample_indices,
        sample_predictions
    ):

        point_idx = int(
            point_idx
        )

        prediction = int(
            prediction
        )


        if prediction not in votes[
            point_idx
        ]:

            votes[
                point_idx
            ][prediction] = 0


        votes[
            point_idx
        ][prediction] += 1


    full_predictions = np.zeros(
        original_count,
        dtype=np.int64
    )


    for i in range(
        original_count
    ):

        if len(votes[i]) == 0:

            # Very unlikely, but keep a
            # safe default.
            full_predictions[i] = 6

        else:

            full_predictions[i] = max(
                votes[i],
                key=votes[i].get
            )


    return (
        xyz.astype(
            np.float32
        ),
        full_predictions
    )


# ============================================================
# BUILD OBJECT BOXES
# ============================================================

def detect_dynamic_objects(
    xyz,
    predictions
):

    dynamic_mask = (
        predictions == 3
    )


    dynamic_points = (
        xyz[
            dynamic_mask
        ]
    )


    if len(dynamic_points) < DBSCAN_MIN_POINTS:

        return []


    point_cloud = (
        o3d.geometry.PointCloud()
    )

    point_cloud.points = (
        o3d.utility.Vector3dVector(
            dynamic_points
        )
    )


    labels = (
        np.asarray(
            point_cloud.cluster_dbscan(
                eps=DBSCAN_EPS,
                min_points=DBSCAN_MIN_POINTS,
                print_progress=False
            )
        )
    )


    if len(labels) == 0:

        return []


    objects = []


    for cluster_id in sorted(
        set(labels)
    ):

        if cluster_id < 0:

            continue


        cluster_points = (
            dynamic_points[
                labels == cluster_id
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


        center = (
            min_bound +
            max_bound
        ) / 2.0


        objects.append(
            {
                "center": center,
                "min": min_bound,
                "max": max_bound,
                "dimensions": dimensions,
                "points": len(
                    cluster_points
                )
            }
        )


    return objects


# ============================================================
# COLOR SEMANTIC POINT CLOUD
# ============================================================

def create_colored_cloud(
    xyz,
    predictions
):

    colors = np.zeros(
        (
            len(xyz),
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


    cloud = (
        o3d.geometry.PointCloud()
    )


    cloud.points = (
        o3d.utility.Vector3dVector(
            xyz
        )
    )


    cloud.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )


    return cloud


# ============================================================
# LiDAR CALLBACK
# ============================================================

def lidar_callback(data):

    global frame_counter

    global latest_points
    global latest_predictions
    global latest_boxes

    global best_points
    global best_predictions
    global best_boxes

    global best_dynamic_objects


    frame_counter += 1


    raw = np.frombuffer(
        data.raw_data,
        dtype=np.float32
    )


    raw = raw.reshape(
        (-1, 4)
    )


    xyz = (
        raw[:, :3]
        .astype(
            np.float32
        )
    )


    # --------------------------------------------------------
    # VALID POINTS
    # --------------------------------------------------------

    valid = np.isfinite(
        xyz
    ).all(axis=1)


    xyz = xyz[
        valid
    ]


    if len(xyz) == 0:

        return


    # --------------------------------------------------------
    # RANGE FILTER
    # --------------------------------------------------------

    distances = np.linalg.norm(
        xyz,
        axis=1
    )


    xyz = xyz[
        distances <=
        LIDAR_RANGE
    ]


    if len(xyz) == 0:

        return


    # --------------------------------------------------------
    # POINTNET INFERENCE
    # --------------------------------------------------------

    xyz, predictions = (
        predict_frame(
            xyz
        )
    )


    # --------------------------------------------------------
    # DETECT DYNAMIC OBJECTS
    # --------------------------------------------------------

    objects = (
        detect_dynamic_objects(
            xyz,
            predictions
        )
    )


    latest_points = xyz
    latest_predictions = predictions
    latest_boxes = objects


    # --------------------------------------------------------
    # SAVE BEST FRAME
    # --------------------------------------------------------

    dynamic_count = len(
        objects
    )


    if (
        dynamic_count
        >= best_dynamic_objects
        and
        len(xyz) > 0
    ):

        best_dynamic_objects = (
            dynamic_count
        )

        best_points = (
            xyz.copy()
        )

        best_predictions = (
            predictions.copy()
        )

        best_boxes = (
            list(objects)
        )


    # --------------------------------------------------------
    # PRINT EVERY ~10 FRAMES
    # --------------------------------------------------------

    if frame_counter % 1 == 0:

        counts = np.bincount(
            predictions,
            minlength=NUM_CLASSES
        )


        print()
        print(
            "--------------------------------------------"
        )

        print(
            "LiDAR Frame:",
            data.frame
        )

        print(
            "Points:",
            len(xyz)
        )

        print(
            "Detected dynamic objects:",
            dynamic_count
        )


        for class_id in range(
            NUM_CLASSES
        ):

            if counts[class_id] > 0:

                print(
                    f"  {class_id}: "
                    f"{CLASS_NAMES[class_id]} "
                    f"→ "
                    f"{counts[class_id]} points"
                )


        for i, obj in enumerate(
            objects
        ):

            c = obj["center"]

            d = obj["dimensions"]


            print(
                f"  Object {i + 1}: "
                f"center=("
                f"{c[0]:.2f}, "
                f"{c[1]:.2f}, "
                f"{c[2]:.2f}) "
                f"size=("
                f"{d[0]:.2f} x "
                f"{d[1]:.2f} x "
                f"{d[2]:.2f}) "
                f"points="
                f"{obj['points']}"
            )


# ============================================================
# START LiDAR
# ============================================================

lidar.listen(
    lidar_callback
)


# ============================================================
# START EGO VEHICLE
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
    "LIVE POINTNET++ TEST STARTED"
)

print(
    "============================================"
)

print(
    "Tesla is driving."
)

print(
    "PointNet++ is classifying live LiDAR."
)

print(
    "Dynamic-object points are clustered."
)

print(
    f"Collection time: "
    f"{COLLECTION_TIME} seconds"
)

print()


# ============================================================
# DRIVE + FOLLOWING CAMERA
# ============================================================

spectator = (
    world.get_spectator()
)


start_time = (
    time.time()
)


while (
    time.time() -
    start_time
    <
    COLLECTION_TIME
):

    vehicle_transform = (
        vehicle.get_transform()
    )


    # Camera behind the Tesla
    camera_location = (
        vehicle_transform.transform(
            carla.Location(
                x=-14.0,
                y=0.0,
                z=6.0
            )
        )
    )


    # Face same direction as Tesla
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


    time.sleep(
        0.05
    )


# ============================================================
# STOP LiDAR
# ============================================================

lidar.stop()

time.sleep(
    0.5
)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print(
    "============================================"
)

print(
    "LIVE TEST COMPLETE"
)

print(
    "============================================"
)

print(
    "LiDAR frames processed:",
    frame_counter
)

print(
    "Best dynamic objects detected:",
    best_dynamic_objects
)


# ============================================================
# VISUALIZATION
# ============================================================

if (
    best_points is not None
    and
    len(best_points) > 0
):

    cloud = (
        create_colored_cloud(
            best_points,
            best_predictions
        )
    )


    geometries = [
        cloud
    ]


    # --------------------------------------------------------
    # BOUNDING BOXES
    # --------------------------------------------------------

    for obj in best_boxes:

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
    # COORDINATE FRAME
    # --------------------------------------------------------

    coordinate_frame = (
        o3d.geometry
        .TriangleMesh
        .create_coordinate_frame(
            size=2.0
        )
    )


    geometries.append(
        coordinate_frame
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    o3d.io.write_point_cloud(
        "carla_pointnet_detection.ply",
        cloud
    )


    with open(
        "carla_pointnet_detections.csv",
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
            best_boxes
        ):

            c = obj["center"]

            d = obj["dimensions"]


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
        "Opening PointNet++ detection visualization..."
    )


    o3d.visualization.draw_geometries(
        geometries,
        window_name=(
            "CARLA + PointNet++ "
            "Semantic Object Detection"
        )
    )


else:

    print()
    print(
        "No valid prediction cloud was captured."
    )


# ============================================================
# CLEANUP
# ============================================================

try:

    lidar.destroy()

except:

    pass


for controller in (
    walker_controllers
):

    try:

        controller.stop()

    except:

        pass


    try:

        controller.destroy()

    except:

        pass


for walker in walkers:

    try:

        walker.destroy()

    except:

        pass


for npc in npc_vehicles:

    try:

        npc.destroy()

    except:

        pass


try:

    vehicle.destroy()

except:

    pass


print()
print(
    "All CARLA actors cleaned up."
)

print(
    "Detection experiment complete."
)