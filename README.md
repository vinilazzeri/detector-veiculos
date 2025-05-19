# 🚗 Vehicle Detection System with YOLOv8

![Project Banner](https://github.com/user-attachments/assets/031f9918-edee-4d72-b4ac-727aaf6f6537)

A complete computer vision solution for detecting 12 vehicle classes with an optimized YOLOv8 model, featuring a Streamlit interface and Docker support.

---

## 📌 Key Features

- 🎯 **High Accuracy Model**: 90.2% Precision | 90.7% Recall | 93.5% mAP50  
- 🖥️ **Interactive Web App**: Image and video processing with real-time visualization  
- 🐳 **Containerized Solution**: Ready-to-deploy Docker container  
- 📊 **Comprehensive Analysis**: Detailed performance metrics and visualizations  
- ⚙️ **Optimized Training**: Custom data augmentation and fine-tuned hyperparameters  

---

## 🚀 Quick Start

### 🔧 Prerequisites

- Python 3.10+  
- NVIDIA GPU (recommended)  
- Docker (for container deployment)  

### 🧪 Installation

```bash
git clone https://github.com/vinilazzeri/detector-veiculos.git
cd detector-veiculos
pip install -r requirements.txt
```

---

## 🖥️ Running the Streamlit App

```bash
streamlit run scripts/app.py
```

This will launch a local web interface where you can upload images or videos and visualize the detection results in real time.

---

## 🐳 Docker Usage

### Build the container

```bash
docker build -t vehicle-detector .
```

### Or use Docker Compose

```bash
docker-compose up --build
```

### Access the app

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔍 Visual Results

Below are sample predictions made by the YOLOv8 model on test images and videos:

| Image 1 | Image 2 | Image 3|
|--------|--------|--------|
| ![Prediction 1](https://github.com/user-attachments/assets/b9dd9402-0fd2-4159-a4c7-16a0a9f6d1ea) | ![Prediction 2](https://github.com/user-attachments/assets/6233730b-fd64-4469-96ed-0b7aa2a28859) | ![Prediction 3](https://github.com/user-attachments/assets/69dc1adf-1254-410f-9182-5bb2c22ade12) |

```markdown
[![Video Preview](imgs/video-thumb.jpg)](https://github.com/vinilazzeri/detector-veiculos/assets/your_video_link)
```

---

## 📁 Project Structure

```bash
detector-veiculos/
├── configs/                   # Dataset config files (e.g., data.yaml)
├── dataset/                   # Train, validation, and test sets
│   ├── train/
│   ├── valid/
│   └── test/
├── imgs/                      # Project images (logo, predictions)
│   └── logo.png
├── scripts/                   # All Python scripts
│   ├── runs/                  # All results from the model training
│   ├── app.py                 # Streamlit interface
│   ├── evaluate_model.py
│   ├── organizer.py
│   ├── predict.py
│   ├── train.py
│   ├── yolo11n.pt
│   └── yolov8m.pt
├── videos/                    # Sample input videos
│   ├── videoteste1.mp4
│   └── videoteste2.mp4
├── Dockerfile                 # Docker build instructions
├── docker-compose.yaml        # Docker Compose setup
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── README.dataset.txt         # Dataset-specific notes
└── README.roboflow.txt        # Roboflow export notes
```

---

## 📚 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Vinicius Lazzeri**  
[GitHub](https://github.com/vinilazzeri) • [LinkedIn](https://www.linkedin.com/in/vinicius-lazzeri/)
