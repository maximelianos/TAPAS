 
import os
import shutil
import subprocess

# Paths
source_dir = "/home/oguz/Desktop/bottles/attempt4/"
target_rgb_dir = "/home/oguz/Desktop/attempt_three/test_data/observe_test/rgb/"
script_path = "/home/oguz/Desktop/attempt_three/my_models/test_keypoint_tapas.py"
output_img = "/home/oguz/Desktop/attempt_three/my_models/keypoints_on_image.jpg"
output_csv = "/home/oguz/Desktop/attempt_three/my_models/keypoints_on_image.csv"

# Iterate over the 20 images
for i in range(20):
    index_str = f"{i:04d}"  # e.g., "0000", "0001", ..., "0019"
    src_img = os.path.join(source_dir, f"{index_str}_color.png")
    dst_img = os.path.join(target_rgb_dir, "00000.jpg")
    
    if not os.path.exists(src_img):
        print(f"Source image not found: {src_img}")
        continue

    # Copy and rename to 00000.jpg
    shutil.copy(src_img, dst_img)
    print(f"Copied {src_img} to {dst_img}")

    try:
        # Run the keypoint extraction script
        subprocess.run(["python3", script_path], check=True)
        print("Script executed successfully.")

        # Define destination names for keypoints
        dst_keypoints_img = os.path.join(source_dir, f"{index_str}_keypoints.jpg")
        dst_keypoints_csv = os.path.join(source_dir, f"{index_str}_keypoints.csv")

        # Copy the output files to the target directory with new names
        shutil.copy(output_img, dst_keypoints_img)
        shutil.copy(output_csv, dst_keypoints_csv)
        print(f"Saved keypoints as {dst_keypoints_img} and {dst_keypoints_csv}\n")

    except subprocess.CalledProcessError as e:
        print(f"Script failed for image {index_str}: {e}")