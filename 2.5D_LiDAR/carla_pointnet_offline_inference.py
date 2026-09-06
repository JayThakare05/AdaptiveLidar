import os
import glob
import csv
import time

import numpy as np
import torch
import open3d as o3d

from train_pointnet2 import PointNet2Segmentation


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "pointnet2_best.pth"

INPUT_DIR = "carla_pointnet_capture"

OUTPUT_DIR = "carla_pointnet_predictions"

NUM_POINTS = 2048

NUM_CLASSES = 7

MAX_FRAMES = 10

DBSCAN_EPS = 1.5

DBSCAN_MIN_POINTS = 8


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
# COLORS
# ============================================================

CLASS_COLORS = {
    0: [0.35, 0.35, 0.35],
    1: [0.65, 0.65, 0.65],
    2: [1.00, 0.10, 0.10],
    3: [1.00, 0.50, 0.00],
    4: [0.10, 0.70, 0.10],
    5: [1.00, 0.85, 0.00],
    6: [0.60, 0.10, 0.70]
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
print("CARLA + POINTNET++ OFFLINE INFERENCE")
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
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n"
        f"{os.path.abspath(MODEL_PATH)}"
    )


if not os.path.exists(INPUT_DIR):

    raise FileNotFoundError(
        f"Capture directory not found:\n"
        f"{os.path.abspath(INPUT_DIR)}"
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(
    "Creating PointNet++ model..."
)

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
# WARM-UP
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


for _ in range(2):

    with torch.inference_mode():

        _ = model(
            warmup_input
        )

    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


del warmup_input


print(
    "Warm-up complete."
)


# ============================================================
# FIND CAPTURED FRAMES
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "frame_*.npz"
        )
    )
)


print()
print(
    "Captured frames found:",
    len(files)
)


if len(files) == 0:

    raise RuntimeError(
        "No .npz files found."
    )


files = files[
    :MAX_FRAMES
]


print(
    "Frames selected for test:",
    len(files)
)


# ============================================================
# PREPROCESS
#
# Matches Jay's PointNet++ pipeline:
#
# XYZ only
# 2048 points
# centroid normalization
# unit-sphere normalization
# ============================================================

def preprocess(
    xyz
):

    n = len(xyz)


    if n == 0:

        return None


    rng = np.random.default_rng(
        42
    )


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
# DYNAMIC OBJECT CLUSTERING
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


        # Reject microscopic noise

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
                "min": min_bound,
                "max": max_bound,
                "center": center,
                "dimensions": dimensions,
                "points": len(
                    cluster_points
                )
            }
        )


    return objects


# ============================================================
# PROCESS FRAMES
# ============================================================

summary_rows = []

best_result = None

best_dynamic_points = -1

total_inference_time = 0.0

successful_frames = 0


for index, file_path in enumerate(files):

    print()
    print(
        "------------------------------------------------------------"
    )

    print(
        f"Processing "
        f"{index + 1}/{len(files)}: "
        f"{os.path.basename(file_path)}"
    )


    # --------------------------------------------------------
    # LOAD FRAME
    # --------------------------------------------------------

    data = np.load(
        file_path
    )


    raw_points = data[
        "points"
    ]


    carla_frame = int(
        data["frame"]
    )


    print(
        "CARLA frame:",
        carla_frame
    )

    print(
        "Raw points:",
        len(raw_points)
    )


    # --------------------------------------------------------
    # XYZ ONLY
    # --------------------------------------------------------

    xyz = (
        raw_points[
            :, :3
        ].astype(
            np.float32
        )
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    valid = np.isfinite(
        xyz
    ).all(
        axis=1
    )


    xyz = xyz[
        valid
    ]


    print(
        "Valid points:",
        len(xyz)
    )


    if len(xyz) == 0:

        continue


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    result = preprocess(
        xyz
    )


    if result is None:

        continue


    sampled_raw = result[0]

    normalized = result[1]


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    input_tensor = (
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

    # --------------------------------------------------------
    # SAVE POINTNET PREDICTIONS FOR LATER MAPPING
    # --------------------------------------------------------

    prediction_file = os.path.join(
        OUTPUT_DIR,
        f"prediction_{index:06d}.npz"
    )

    np.savez_compressed(
        prediction_file,
        points=sampled_raw,
        predictions=predictions,
        frame=np.array(
            carla_frame,
            dtype=np.int64
        )
    )

    total_inference_time += (
        inference_time
    )

    successful_frames += 1


    # --------------------------------------------------------
    # SEMANTIC DISTRIBUTION
    # --------------------------------------------------------

    counts = np.bincount(
        predictions,
        minlength=NUM_CLASSES
    )


    dynamic_points = int(
        counts[3]
    )


    # --------------------------------------------------------
    # OBJECT CLUSTERS
    # --------------------------------------------------------

    objects = (
        detect_dynamic_objects(
            sampled_raw,
            predictions
        )
    )


    print()
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
        "Predicted classes:"
    )


    for class_id in range(
        NUM_CLASSES
    ):

        if counts[class_id] > 0:

            print(
                f"  {class_id}: "
                f"{CLASS_NAMES[class_id]:<18}"
                f"→ "
                f"{counts[class_id]}"
            )


    # --------------------------------------------------------
    # OBJECT OUTPUT
    # --------------------------------------------------------

    for object_id, obj in enumerate(
        objects
    ):

        c = obj["center"]

        d = obj["dimensions"]


        print(
            f"  Dynamic Object "
            f"{object_id + 1}: "
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
    # SUMMARY
    # --------------------------------------------------------

    summary_rows.append(
        [
            index,
            carla_frame,
            len(raw_points),
            len(xyz),
            inference_time,
            dynamic_points,
            len(objects)
        ]
    )


    # --------------------------------------------------------
    # BEST FRAME
    # --------------------------------------------------------

    if dynamic_points > (
        best_dynamic_points
    ):

        best_dynamic_points = (
            dynamic_points
        )


        best_result = {
            "file":
                file_path,

            "frame":
                carla_frame,

            "points":
                sampled_raw.copy(),

            "predictions":
                predictions.copy(),

            "objects":
                list(objects),

            "inference_time":
                inference_time
        }


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "inference_summary.csv"
)


with open(
    summary_file,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow(
        [
            "index",
            "carla_frame",
            "raw_points",
            "valid_points",
            "inference_time",
            "dynamic_points",
            "dynamic_clusters"
        ]
    )


    writer.writerows(
        summary_rows
    )


# ============================================================
# FINAL STATISTICS
# ============================================================

print()
print("=" * 70)
print("OFFLINE INFERENCE COMPLETE")
print("=" * 70)

print(
    "Frames processed:",
    successful_frames
)


if successful_frames > 0:

    print(
        "Average inference time:",
        f"{total_inference_time / successful_frames:.4f}s"
    )


print(
    "Summary saved:",
    summary_file
)


# ============================================================
# FINAL VISUALIZATION
# ============================================================

if best_result is None:

    print(
        "No successful inference result."
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
        best_dynamic_points
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
    # POINT CLOUD
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
    # DYNAMIC OBJECT BOXES
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


    print()
    print(
        "Opening best CARLA PointNet++ result..."
    )


    o3d.visualization.draw_geometries(
        geometries,
        window_name=(
            "CARLA LiDAR + "
            "PointNet++ Offline Inference"
        )
    )


print()
print(
    "Experiment finished."
)