# Hand-Gesture-Frame
pake ini biar gak cape ngedit

# Hand Gesture Frame Tracking

Project computer vision berbasis Python yang menggunakan OpenCV dan MediaPipe untuk mendeteksi gerakan tangan secara real-time melalui webcam.

Sistem melacak posisi jari telunjuk dan jempol dari dua tangan untuk membentuk sebuah frame. Area di dalam frame diberikan efek filter gradient warna yang mengikuti pergerakan tangan.

## Features

- Real-time hand tracking
- Deteksi jempol dan jari telunjuk
- Frame mengikuti gerakan tangan
- Filter gradient 4 warna
- Efek transparan
- Webcam processing menggunakan OpenCV

## Technologies

- Python
- OpenCV
- MediaPipe 1.0.1
- NumPy

## Project Structure

```text
hand-gesture-frame/
├── main.py
├── hand_landmarker.task
├── requirements.txt
└── README.md
