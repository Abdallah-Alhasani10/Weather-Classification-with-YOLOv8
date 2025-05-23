
# Weather Classification with YOLOv8

This project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for image classification to detect weather conditions from images (e.g., cloudy, rain, shine, sunrise).

## 📁 Dataset Structure

Ensure your dataset follows this structure:

```
weather/
├── train/
│   ├── cloudy/
│   ├── rain/
│   ├── shine/
│   └── sunrise/
├── val/
│   ├── cloudy/
│   ├── rain/
│   ├── shine/
│   └── sunrise/
```

Each folder should contain images corresponding to that weather condition.

## 🚀 Installation

1. Clone this repository:

```bash
git clone https://github.com/Abdallah-Alhasani10/Weather-Classification-with-YOLOv8.git
cd weather-yolov8
```

2. Install dependencies:

```bash
pip install ultralytics numpy
```

## 🏋️‍♂️ Training

To train the YOLOv8 classification model on your dataset:

```python
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # Load the pre-trained classification model
results = model.train(data="./weather", epochs=7, imgsz=64)
```

> Make sure `./weather` contains `train` and `val` subdirectories with class folders inside.

## 🔍 Inference

To make a prediction on a new image:

```python
from ultralytics import YOLO
import numpy as np

model = YOLO('./runs/classify/train/weights/best.pt')
res = model('./download.jpeg')

names_dict = res[0].names
probs = res[0].probs.data.tolist()

print(names_dict[np.argmax(probs)])  # Output the predicted class
```

## 📦 Output

The model will print the predicted weather class (e.g., `cloudy`, `rain`, `shine`, `sunrise`) for the input image.

## 📄 License

This project is open-source under the MIT license.

## 🤝 Contributing

Feel free to submit pull requests or open issues to improve the project.
