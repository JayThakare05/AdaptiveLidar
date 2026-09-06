import os
import random
import time

import numpy as np
import carla


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "carla_pointnet_capture"

COLLECTION_TIME = 20.0

NUM_BACKGROUND_VEHICLES = 10
NUM_PEDESTRIANS = 15


# ============================================================
# LiDAR CONFIGURATION
#
# Same configuration currently being used for the
# PointNet/CARLA experiment.
# ============================================================

LIDAR_RANGE = 50.0

LIDAR_CHANNELS = 32

LIDAR_POINTS_PER_SECOND = 130000

LIDAR_ROTATION_FREQUENCY = 10.0

LIDAR_SENSOR_TICK = 0.1


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
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
print("=" * 70)
print("CARLA LiDAR DATASET CAPTURE")
print("=" * 70)

print(
    "Map:",
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


ego_location = (
    vehicle.get_location()
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


vehicle_spawn_points = (
    world.get_map().get_spawn_points()
)


safe_spawns = []


for sp in vehicle_spawn_points:

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
    "LiDAR configuration:"
)

print(
    f"  Channels: {LIDAR_CHANNELS}"
)

print(
    f"  Points/sec: {LIDAR_POINTS_PER_SECOND}"
)

print(
    f"  Range: {LIDAR_RANGE} m"
)

print(
    f"  Rotation: {LIDAR_ROTATION_FREQUENCY} Hz"
)

print(
    f"  Sensor tick: {LIDAR_SENSOR_TICK} s"
)


# ============================================================
# DATA STORAGE
# ============================================================

frame_count = 0

total_points = 0

first_frame = None

last_frame = None

capture_start = time.time()


# ============================================================
# LIDAR CALLBACK
# ============================================================

def lidar_callback(data):

    global frame_count
    global total_points
    global first_frame
    global last_frame


    raw = np.frombuffer(
        data.raw_data,
        dtype=np.float32
    )


    raw = raw.reshape(
        (-1, 4)
    )


    # raw format:
    #
    # X
    # Y
    # Z
    # intensity
    #
    # Keep all four channels because this
    # matches the dataset representation.


    valid = np.isfinite(
        raw
    ).all(
        axis=1
    )


    points = (
        raw[
            valid
        ].astype(
            np.float32
        )
    )


    if len(points) == 0:

        return


    # --------------------------------------------------------
    # Save frame
    # --------------------------------------------------------

    frame_index = frame_count


    output_file = os.path.join(
        OUTPUT_DIR,
        f"frame_{frame_index:06d}.npz"
    )


    np.savez_compressed(
        output_file,
        points=points,
        frame=np.array(
            data.frame,
            dtype=np.int64
        )
    )


    frame_count += 1

    total_points += len(points)


    if first_frame is None:

        first_frame = data.frame


    last_frame = data.frame


    print(
        f"\rCaptured frame "
        f"{frame_count:04d} | "
        f"CARLA frame "
        f"{data.frame} | "
        f"points "
        f"{len(points):5d}",
        end=""
    )


# ============================================================
# START LIDAR
# ============================================================

lidar.listen(
    lidar_callback
)


# ============================================================
# START TESLA
# ============================================================

vehicle.set_autopilot(
    True,
    traffic_manager.get_port()
)


# ============================================================
# CAMERA
# ============================================================

spectator = (
    world.get_spectator()
)


print()
print()
print("=" * 70)

print(
    "CARLA DATA CAPTURE STARTED"
)

print("=" * 70)

print(
    "Tesla is driving."
)

print(
    "Traffic is active."
)

print(
    "LiDAR frames are being saved."
)

print(
    "PointNet++ is NOT running."
)

print(
    f"Collection time: "
    f"{COLLECTION_TIME} seconds"
)

print()


# ============================================================
# COLLECTION LOOP
# ============================================================

try:

    while (
        time.time()
        -
        capture_start
        <
        COLLECTION_TIME
    ):

        # ----------------------------------------------------
        # FOLLOW TESLA
        # ----------------------------------------------------

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


        time.sleep(
            0.02
        )


finally:

    try:

        lidar.stop()

    except Exception:

        pass


# ============================================================
# CAPTURE COMPLETE
# ============================================================

print()
print()
print("=" * 70)

print(
    "CARLA DATA CAPTURE COMPLETE"
)

print("=" * 70)

print(
    "Frames captured:",
    frame_count
)

print(
    "Total points:",
    total_points
)


if frame_count > 0:

    print(
        "Average points/frame:",
        f"{total_points / frame_count:.1f}"
    )

    print(
        "First CARLA frame:",
        first_frame
    )

    print(
        "Last CARLA frame:",
        last_frame
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
    "Saved dataset directory:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print()

print(
    "Capture experiment complete."
)