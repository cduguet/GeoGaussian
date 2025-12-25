import json
import os
import shutil
import numpy as np
from PIL import Image
import math
import open3d as o3d

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion (w, x, y, z) to a rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def main():
    data_dir = "/data/extracted"
    output_dir = "/home/azureuser/GeoGaussian/data/RealData"
    
    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "results"))
    
    # Read poses
    with open(os.path.join(data_dir, "poses.json"), 'r') as f:
        poses_data = json.load(f)
        
    # Copy and downsample point cloud
    print("Reading and downsampling point cloud...")
    pcd = o3d.io.read_point_cloud(os.path.join(data_dir, "points.ply"))
    
    # Downsample to reduce initial points (e.g. 5M to 100k for OOM avoidance)
    # Target around 100k points for initialization if original is huge (38M)
    target_points = 100000
    points = np.asarray(pcd.points)
    if len(points) > target_points:
        # Random sampling
        indices = np.random.choice(len(points), size=target_points, replace=False)
        pcd_down = pcd.select_by_index(indices)
        print(f"Downsampled from {len(points)} to {len(indices)}")
    else:
        pcd_down = pcd
        print(f"Kept {len(points)} points")
        
    o3d.io.write_point_cloud(os.path.join(output_dir, "PointCloud.ply"), pcd_down)
    
    # Process frames
    image_files = sorted(poses_data.keys())
    
    trajectory_file = os.path.join(output_dir, "KeyFrameTrajectory2.txt")
    
    with open(trajectory_file, 'w') as f:
        for i, img_file in enumerate(image_files):
            if i % 10 == 0:
                print(f"Processing frame {i}/{len(image_files)}")
            # Copy image
            src_img = os.path.join(data_dir, "images", img_file)
            
            dst_img_name = f"frame{i:06d}.jpg"
            dst_img = os.path.join(output_dir, "results", dst_img_name)
            
            # Convert to jpg if necessary, or just copy
            img = Image.open(src_img)
            # Ensure RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize image to avoid OOM? GeoGaussian resizes > 1.6K.
            # Let's keep original for now, but maybe resize if huge.
            img.save(dst_img)
            
            # Get pose
            pose_data = poses_data[img_file]
            trans = pose_data['translation']
            rot = pose_data['rotation_quaternion'] # [w, x, y, z]
            
            R = quaternion_to_rotation_matrix(rot)
            
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = trans
            
            # Write to file
            # Format: index r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz 0 0 0 1
            flat_c2w = c2w.flatten()
            line = f"{i} " + " ".join(map(str, flat_c2w))
            f.write(line + "\n")
            
    print(f"Prepared data at {output_dir}")

if __name__ == "__main__":
    main()