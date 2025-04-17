import cv2
import mediapipe as mp
import numpy as np

# List of T-shirt/dress image paths
tshirt_paths = [
    'C:\\Users\\vinut\\OneDrive\\virtual_tryon\\blackbg.png',
    'C:\\Users\\vinut\\OneDrive\\virtual_tryon\\fancy.png',
    'C:\\Users\\vinut\\OneDrive\\virtual_tryon\\pink1.png'
]

# Function to remove both white and black backgrounds
def remove_background(img, threshold=30):
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    white_mask = np.all(img[:, :, :3] >= (255 - threshold), axis=2)
    black_mask = np.all(img[:, :, :3] <= threshold, axis=2)

    alpha_mask = ~(white_mask | black_mask)
    img_bgra[:, :, 3] = (alpha_mask * 255).astype(np.uint8)

    return img_bgra

# Load initial T-shirt
current_index = 0
tshirt_img = cv2.imread(tshirt_paths[current_index])
tshirt_img = remove_background(tshirt_img)
has_alpha = True

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Webcam
cap = cv2.VideoCapture(0)

# For smoothing movement
prev_x, prev_y = 0, 0

# Function to rotate image
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        x1 = int(left_shoulder.x * w)
        y1 = int(left_shoulder.y * h)
        x2 = int(right_shoulder.x * w)
        y2 = int(right_shoulder.y * h)
        x3 = int(left_hip.x * w)
        y3 = int(left_hip.y * h)
        x4 = int(right_hip.x * w)
        y4 = int(right_hip.y * h)

        # Calculate size based on shoulders and hips
        tshirt_width = int(1.4 * abs(x2 - x1))
        torso_height = max(abs(y3 - y1), abs(y4 - y2))
        tshirt_height = int(1.6 * torso_height)

        # Shoulder angle
        shoulder_angle = np.arctan2(y1 - y2, x1 - x2) * 180 / np.pi

        # Center of the dress
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y3) / 2)

        # Smooth movement
        x_center = int(0.7 * prev_x + 0.3 * x_center)
        y_center = int(0.7 * prev_y + 0.3 * y_center)
        prev_x, prev_y = x_center, y_center

        x_start = int(x_center - tshirt_width / 2)
        y_start = int(y_center - tshirt_height / 2)

        # Resize and rotate the dress
        resized_tshirt = cv2.resize(tshirt_img, (tshirt_width, tshirt_height))
        rotated_tshirt = rotate_image(resized_tshirt, shoulder_angle)

        # Overlay on the frame
        for i in range(tshirt_height):
            for j in range(tshirt_width):
                if 0 <= x_start + j < w and 0 <= y_start + i < h:
                    alpha = rotated_tshirt[i, j, 3] / 255.0
                    for c in range(3):
                        frame[y_start + i, x_start + j, c] = int(
                            alpha * rotated_tshirt[i, j, c] + (1 - alpha) * frame[y_start + i, x_start + j, c]
                        )

    cv2.imshow('Virtual Clothes Try-On', frame)

    # Keypress events
    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        current_index = 0
    elif key == ord('2'):
        current_index = 1
    elif key == ord('3'):
        current_index = 2

    if key in [ord('1'), ord('2'), ord('3')]:
        tshirt_img = cv2.imread(tshirt_paths[current_index])
        tshirt_img = remove_background(tshirt_img)
        has_alpha = True

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
