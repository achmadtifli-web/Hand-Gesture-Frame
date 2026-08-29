import cv2
import mediapipe as mp
import numpy as np
import math
import os

MODEL_PATH = "hand_landmarker.task"

CAMERA_INDEX = 0

MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

MAX_HANDS = 2

# cek model

if not os.path.exists(MODEL_PATH):
    print("=" * 60)
    print("ERROR: hand_landmarker.task tidak ditemukan!")
    print("=" * 60)
    print()
    print("Letakkan file:")
    print("hand_landmarker.task")
    print()
    print("di folder yang sama dengan main.py")
    print()
    exit()

# mediapipe task

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

RunningMode = mp.tasks.vision.RunningMode

# konfig handland maker

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

# fungsi mengambil titik


def get_point(hand_landmarks, landmark_id, width, height):

    landmark = hand_landmarks[landmark_id]

    x = int(landmark.x * width)

    y = int(landmark.y * height)

    return (x, y)


# jarak antara 2 titik


def calculate_distance(point1, point2):

    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# draw tracking point


def draw_tracking_point(frame, point, color):

    x, y = point

    # Lingkaran luar
    cv2.circle(frame, (x, y), 12, color, 2)

    # Titik tengah
    cv2.circle(frame, (x, y), 5, color, -1)


# draw label


def draw_label(frame, point, text, color):

    x, y = point

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.45

    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    padding = 8

    box_x1 = x + 15

    box_y1 = y - text_height - 15

    box_x2 = box_x1 + text_width + padding * 2

    box_y2 = box_y1 + text_height + padding * 2

    # Jika keluar kanan
    if box_x2 > frame.shape[1]:

        box_x2 = x - 15

        box_x1 = box_x2 - text_width - padding * 2

    # Jika keluar atas
    if box_y1 < 0:

        box_y1 = y + 15

        box_y2 = box_y1 + text_height + padding * 2

    # Background transparan
    overlay = frame.copy()

    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)

    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Border
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, 1)

    # Text
    cv2.putText(
        frame,
        text,
        (box_x1 + padding, box_y2 - padding),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# draw color


def draw_filter(frame, points, alpha=0.30):

    height, width = frame.shape[:2]

    # bounding box
    min_x = max(0, min(p[0] for p in points))
    max_x = min(width - 1, max(p[0] for p in points))

    min_y = max(0, min(p[1] for p in points))
    max_y = min(height - 1, max(p[1] for p in points))

    if min_x >= max_x or min_y >= max_y:
        return

    # ukuran area

    w = max_x - min_x + 1
    h = max_y - min_y + 1

    x = np.linspace(0, 1, w, dtype=np.float32)

    y = np.linspace(0, 1, h, dtype=np.float32)

    X, Y = np.meshgrid(x, y)

    # warna 4 sudut untuk gradien warna

    top_left = np.array([255, 255, 0], dtype=np.float32)

    top_right = np.array([255, 0, 255], dtype=np.float32)

    bottom_left = np.array([255, 0, 0], dtype=np.float32)

    bottom_right = np.array([255, 0, 180], dtype=np.float32)

    # gradiant warnanya

    top = (
        top_left[None, None, :] * (1 - X[:, :, None])
        + top_right[None, None, :] * X[:, :, None]
    )

    bottom = (
        bottom_left[None, None, :] * (1 - X[:, :, None])
        + bottom_right[None, None, :] * X[:, :, None]
    )

    gradient = top * (1 - Y[:, :, None]) + bottom * Y[:, :, None]

    gradient = gradient.astype(np.uint8)

    # mask polygon

    mask = np.zeros((h, w), dtype=np.uint8)

    polygon = np.array([[p[0] - min_x, p[1] - min_y] for p in points], dtype=np.int32)

    cv2.fillPoly(mask, [polygon], 255)

    # ambil area frame

    roi = frame[min_y : max_y + 1, min_x : max_x + 1]

    # blend hanya bagian polygon  

    mask_bool = mask == 255

    roi[mask_bool] = (
        roi[mask_bool] * (1 - alpha) + gradient[mask_bool] * alpha
    ).astype(np.uint8)


# draw frame

def draw_frame(frame, left_index, right_index, left_thumb, right_thumb):

    # urutan titik:

    points = np.array(
        [left_index, right_index, right_thumb, left_thumb], dtype=np.int32
    )

    # frame utama
    cv2.polylines(frame, [points], True, (0, 255, 255), 3, cv2.LINE_AA)

    # garis tambahan tipis
    cv2.polylines(frame, [points], True, (255, 255, 255), 1, cv2.LINE_AA)

# webcam

cap = cv2.VideoCapture(CAMERA_INDEX)


if not cap.isOpened():

    print("ERROR: Kamera tidak dapat dibuka.")

    exit()


# Resolusi kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow(
    "Live Hand Gesture Frame Tracking",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Live Hand Gesture Frame Tracking",
    640,
    480
)

# timer

timestamp_ms = 0

# main

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:

            print("Gagal membaca kamera.")

            break

        # Mirror kamera
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp_ms += 33

        # deteksi tangan

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        hands_data = []

        for i, hand_landmarks in enumerate(result.hand_landmarks):

            # Handedness
            handedness = result.handedness[i][0]

            hand_label = handedness.category_name

            hand_score = handedness.score

            # Wrist
            wrist = get_point(hand_landmarks, 0, width, height)

            # Simpan data
            hands_data.append(
                {
                    "landmarks": hand_landmarks,
                    "label": hand_label,
                    "score": hand_score,
                    "wrist": wrist,
                }
            )

        hands_data.sort(key=lambda hand: hand["wrist"][0])

        # 2 jari terdeteksi

        if len(hands_data) == 2:

            left_hand = hands_data[0]

            right_hand = hands_data[1]

            left_landmarks = left_hand["landmarks"]

            right_landmarks = right_hand["landmarks"]

            left_thumb = get_point(left_landmarks, 4, width, height)

            left_index = get_point(left_landmarks, 8, width, height)

            right_thumb = get_point(right_landmarks, 4, width, height)

            right_index = get_point(right_landmarks, 8, width, height)

            # warna

            YELLOW = (0, 255, 255)

            CYAN = (255, 255, 0)

            GREEN = (0, 255, 0)

            WHITE = (255, 255, 255)

            draw_tracking_point(frame, left_index, YELLOW)

            draw_tracking_point(frame, right_index, YELLOW)

            draw_tracking_point(frame, left_thumb, CYAN)

            draw_tracking_point(frame, right_thumb, CYAN)

            draw_label(frame, left_index, "INDEX", YELLOW)

            draw_label(frame, right_index, "INDEX", YELLOW)

            draw_label(frame, left_thumb, "THUMB", CYAN)

            draw_label(frame, right_thumb, "THUMB", CYAN)

            # frame
            points = [left_index, right_index, right_thumb, left_thumb]

            draw_filter(frame, points, alpha=0.45)

            draw_frame(frame, left_index, right_index, left_thumb, right_thumb)

            left_distance = calculate_distance(left_thumb, left_index)

            right_distance = calculate_distance(right_thumb, right_index)

            pinch_threshold = 55

            left_pinch = left_distance < pinch_threshold

            right_pinch = right_distance < pinch_threshold

            cv2.putText(
                frame,
                "HAND GESTURE FRAME",
                (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                GREEN,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "2 HANDS TRACKING",
                (25, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                WHITE,
                1,
                cv2.LINE_AA,
            )

            if left_pinch:

                cv2.putText(
                    frame,
                    "LEFT PINCH",
                    (25, height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    CYAN,
                    2,
                    cv2.LINE_AA,
                )

            if right_pinch:

                cv2.putText(
                    frame,
                    "RIGHT PINCH",
                    (25, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    CYAN,
                    2,
                    cv2.LINE_AA,
                )

        elif len(hands_data) == 1:

            hand = hands_data[0]

            landmarks = hand["landmarks"]

            thumb = get_point(landmarks, 4, width, height)

            index = get_point(landmarks, 8, width, height)

            YELLOW = (0, 255, 255)

            CYAN = (255, 255, 0)

            draw_tracking_point(frame, thumb, CYAN)

            draw_tracking_point(frame, index, YELLOW)

            draw_label(frame, thumb, "THUMB", CYAN)

            draw_label(frame, index, "INDEX", YELLOW)

            cv2.putText(
                frame,
                "1 HAND DETECTED",
                (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                YELLOW,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "SHOW BOTH HANDS",
                (25, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        else:

            cv2.putText(
                frame,
                "SHOW BOTH HANDS",
                (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "Q = Quit",
            (width - 110, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Live Hand Gesture Frame Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

cap.release()

cv2.destroyAllWindows()

print("Program selesai.")
