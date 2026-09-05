import open3d as o3d
import os


# ============================================================
# LOAD PLY FILE
# ============================================================

filename = "carla_lidar_map.ply"

print("Looking for:", os.path.abspath(filename))

if not os.path.exists(filename):
    print("ERROR: PLY file not found.")
    exit()


# ============================================================
# READ POINT CLOUD
# ============================================================

pcd = o3d.io.read_point_cloud(filename)

print()
print("============================================")
print("PLY FILE LOADED")
print("============================================")

print("Number of points:", len(pcd.points))
print("Has colors:", pcd.has_colors())
print("Has normals:", pcd.has_normals())


# ============================================================
# VISUALIZE
# ============================================================

print()
print("Opening point cloud...")

o3d.visualization.draw_geometries(
    [pcd],
    window_name="CARLA LiDAR Map - Loaded from PLY"
)


print("Done.")