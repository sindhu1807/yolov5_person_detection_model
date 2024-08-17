import torch
import cv2
import matplotlib.pyplot as plt
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the pre-trained small model

# Use relative paths for the dataset
base_dir = "./dataset"
output_dir = "./detections"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to process images in each subfolder
def process_images_in_folder(folder_path):
    # List all subfolders (e.g., 'images', 'labels') and images
    image_folder = os.path.join(folder_path, 'images')
    
    if not os.path.exists(image_folder):
        print(f"No 'images' folder found in {folder_path}")
        return
    
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        # Check if the image is loaded properly
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        
        # Perform inference
        results = model(image)
        
        # Save results
        result_save_path = os.path.join(output_dir, os.path.basename(folder_path))
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)
        
        # Save results
        results.save(result_save_path)
        
        # Display results using Matplotlib
        if results.ims:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(results.ims[0], cv2.COLOR_BGR2RGB))
            plt.title(f'Detections for {image_file}')
            plt.axis('off')  # Hide axis
            plt.show()
        else:
            print(f"No results images found for {image_file}")

# Process each subfolder in the base dataset directory
for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder)
    
    if os.path.isdir(subfolder_path):
        process_images_in_folder(subfolder_path)

print("Person detection completed and results saved.")
