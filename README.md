********🧠 Real-Time Face Detection and Classification Pipeline********

A robust deep learning-based system that seamlessly combines **object detection**, **image classification**, and **multiprocessing** for efficient real-time inference. Built using **Python, OpenCV, TensorFlow/Keras**, and optimized with  .tflite model quantization.


****📌 Project Overview****

This project showcases an integrated modular pipeline capable of real-time face detection and classification on video streams. It leverages frame optimization techniques, dynamic label mapping, and parallel processing to achieve fast and scalable predictions.


**🔍 Key Components**

**📁 Instance Attributes**

objdet_path:
Path to the object detection model file (e.g., haarcascade_frontalface.xml).

imgcls_path:
Path to the image classification model file (e.g., .h5, .keras, or .tflite).

**⚙️ Core Methods**

load_objectdetection_model():
Loads the object detection model from the specified objdet_path.

load_imageclassification_model():
Loads the image classification model from the specified imgcls_path.

load_database():
Loads a JSON-based database (db.json) to map predicted class indices to human-readable labels.

process_video(source_path, objectdetmodel, imgclassmodel, db):
Processes video frame-by-frame:

Detects faces using object detection.

Classifies each detected face.

Displays annotated results in real time.

Includes preprocessing steps such as:

Frame downsampling

Grayscale conversion

FPS limiting (15 FPS)

run(video_path, model1, model2, database):
Wrapper function to execute process_video() with all required components.

**🔄 Version-wise Development**

**✅ Version 1: Initial Pipeline Structure**

Designed base class with modular structure.

Implemented instance attributes and core methods:
load_objectdetection_model(), load_imageclassification_model(), process_video(), run().

**🔄 Version 2: Object Detection Integration**

Integrated the object detection model into the pipeline.

Initial latency observed when performing face detection on video frames.

**⚙️ Version 3: Performance Optimization**

Downsampled frame resolution by 50%.

Converted frames to grayscale before detection.

Limited FPS to 15.

Result: Significant improvement in frame processing speed.

**🧠 Version 4: Image Classification Integration**

Integrated image classification model for detected faces.

Issue: Additional latency due to heavy Keras model.

Solution:

Quantized the model to .tflite.

Converted weights to float16 precision.

**🔗 Version 5: Database Mapping**

Mapped classification output indices to custom labels via db.json.

Latency slightly increased due to reintroduction of Keras model.

Next step: Further optimize classification model.

**🚀 Version 6: Final Optimization with Quantization**

Applied Post-Training Quantization (float16) using TensorFlow Lite.

Converted the Keras model to an optimized .tflite format.

Result: Smooth real-time inference with accurate classification.

**🧵 Version 7: Multiprocessing for Concurrent Video Inference**

Integrated Python multiprocessing to handle multiple video streams simultaneously.

Used multiprocessing.Process to run parallel classification tasks.

Result: Dramatic speedup in multi-video scenarios; improved throughput and scalability.

**🛠️ Tech Stack**

**Programming Language**: Python

**Computer Vision**: OpenCV

**Deep Learning:** TensorFlow, Keras

**Model Optimization:** TensorFlow Lite, Float16 quantization

**Parallelism:** Python Multiprocessing

**Data Format:** JSON (for label mapping)

**📁 Folder Structure**

├── objdet_weights
│   └── haarcascade_frontalface.xml      # Pretrained face detection model
│
├── imgcls_weights
│   ├── mobnetv3.keras                   # Quantized MobileNetV3 classification model
│   ├── vgg19.keras
│   └── resnet.keras
│
├── videos
│   └── trimmed.mp4                      # Sample input video(s)
│
├── db.json                              # Class label mapping file
│
├── module.py                            # Core class with pipeline logic
│
├── run.py                               # Entry script for executing the pipeline with multiprocessing
│
└── README.md                            # Project documentation
