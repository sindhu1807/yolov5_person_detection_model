## **1. YOLOv5 Person Detection – `README.md`**

```markdown
# YOLOv5 Batch Person Detection from Dataset

This project implements automated person detection using the YOLOv5 object detection model. It processes a structured dataset of images, applies detection using a pre-trained YOLOv5 model, and saves the output images with bounding boxes.

## Dataset Structure

The dataset should follow this folder structure:

```

dataset/
├── subfolder1/
│   └── images/
│       ├── image1.jpg
│       └── image2.jpg
├── subfolder2/
│   └── images/
│       ├── image3.jpg
│       └── image4.jpg

````

## Technologies Used

- Python 3.x
- PyTorch
- OpenCV
- Matplotlib
- YOLOv5 via Torch Hub

## Usage Instructions

1. Install the required packages:
```bash
pip install torch torchvision opencv-python matplotlib
````

2. Place your dataset in the `./dataset` folder.

3. Run the script:

```bash
python detect_from_dataset.py
```

4. Detected images with bounding boxes will be saved in the `./detections` folder, organized by subfolder.

## Optional: Detect Only People

To restrict detection to only persons (class 0 in the COCO dataset), modify the model as follows in the script:

```python
model.classes = [0]
```

## Output

The script saves the annotated images and optionally displays them using Matplotlib. You can include a sample output image in your repository to demonstrate results.

