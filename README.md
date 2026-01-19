🚗 YOLO + ByteTrack Vehicle Tracking

A real-time vehicle detection and tracking system using YOLO (You Only Look Once) for object detection and ByteTrack for multi-object tracking. The project assigns unique IDs to vehicles across video frames and maintains tracking even in crowded scenes and partial occlusions.

📌 Features

✅ Real-time vehicle detection and tracking

✅ Unique ID assignment for each vehicle

✅ Handles occlusion and overlapping objects

✅ Supports video input and webcam feed

✅ Suitable for traffic analysis and smart surveillance

🏗️ Tech Stack

Python

YOLOv8 (or YOLOv7/YOLOv5 – adjust as per your implementation)

ByteTrack

OpenCV

NumPy

📂 Project Structure (Example)
yolo-bytetrack-vehicle-tracking-main/
│── models/
│── trackers/
│── videos/
│── output/
│── main.py
│── requirements.txt
│── README.md


(You can modify this based on your actual folder structure.)

⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/yolo-bytetrack-vehicle-tracking-main.git
cd yolo-bytetrack-vehicle-tracking-main

2. Install dependencies
pip install -r requirements.txt

▶️ Usage
Run on a video file:
python main.py --source videos/traffic.mp4

Run on webcam:
python main.py --source 0

📊 Applications

Traffic Monitoring

Vehicle Counting

Smart City Surveillance

Accident Detection

Highway Analysis

📌 Future Enhancements

Add speed estimation

Vehicle type classification

License plate recognition

Integration with dashboard

👨‍💻 Author

Annamnedi Govardhan
LinkedIn: www.linkedin.com/in/govardhan-annamnedi-649169243
