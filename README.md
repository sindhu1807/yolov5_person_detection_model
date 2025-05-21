YOLOv5 Batch Person Detection from Dataset

This project implements **automated person detection** using the YOLOv5 object detection model. It scans through a structured dataset of images, performs detection using a pre-trained YOLOv5 model, and saves the output with bounding boxes.

📁 Dataset Structure

Make sure your dataset is structured like this:


dataset/
├── subfolder1/
│   └── images/
│       ├── image1.jpg
│       └── image2.jpg
├── subfolder2/
│   └── images/
│       ├── image3.jpg
│       └── image4.jpg


 🛠️ Tech Stack

- Python 3.x
- PyTorch
- OpenCV
- YOLOv5 (via Torch Hub)
- Matplotlib

🚀 How to Use

1. **Install the dependencies**
```bash
pip install torch torchvision opencv-python matplotlib
````

2. **Run the detection script**

```bash
python detect_from_dataset.py
```

> Make sure your dataset is in the `./dataset` folder, and the script is in the same root directory.

3. **Results**

* Annotated images will be saved in the `./detections` folder, under their respective subfolder names.
* Detected images will include bounding boxes around people and other objects (unless modified to detect only persons).

## 📌 Optional: Detect Only People

If you'd like to restrict the model to detect only persons, modify the script with:

```python
model.classes = [0]  # Class 0 corresponds to 'person' in the COCO dataset
```

## ✨ Output Example

Each image will be saved with YOLOv5-generated bounding boxes and shown via Matplotlib.


## 🧠 What the Script Does

* Loads YOLOv5s via PyTorch Hub
* Recursively scans each subfolder in `./dataset`
* Applies object detection on each image
* Saves the detection outputs to `./detections/[subfolder]/`
* Displays results using Matplotlib
