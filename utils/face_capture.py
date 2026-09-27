import os
import re
import cv2
import numpy as np
from utils.face_detection import FaceDetector


def get_dataset_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir


def sanitize_name(name: str) -> str:
    clean = re.sub(r'[^a-zA-Z0-9_\- ]', '', name).strip()
    return clean.replace(' ', '_')


def sanitize_id(person_id: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]', '', str(person_id)).strip()


def check_existing_id(person_id: str) -> tuple[bool, str]:
    dataset_dir = get_dataset_dir()
    person_id = sanitize_id(person_id)
    if not os.path.exists(dataset_dir):
        return False, ""

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            parts = folder.rsplit('_', 1)
            if len(parts) == 2 and parts[1] == person_id:
                return True, folder
            if folder == person_id:
                return True, folder
    return False, ""


def create_person_directory(person_name: str, person_id: str) -> str:
    dataset_dir = get_dataset_dir()
    clean_name = sanitize_name(person_name)
    clean_id = sanitize_id(person_id)
    folder_name = f"{clean_name}_{clean_id}"
    person_dir = os.path.join(dataset_dir, folder_name)
    os.makedirs(person_dir, exist_ok=True)
    return person_dir


def test_working_camera() -> int:
    for idx in (0, 1, 2):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                return idx
    return -1


def evaluate_face_quality(gray_img, face_rect) -> tuple[bool, str, float]:
    """Evaluates face image quality based on size, sharpness (Laplacian variance), and contrast.
    Returns (is_valid, quality_status_msg, sharpness_score).
    """
    x, y, w, h = face_rect
    if w < 60 or h < 60:
        return False, "Too Far / Move Closer", 0.0

    face_roi = gray_img[y:y+h, x:x+w]
    if face_roi.size == 0:
        return False, "Invalid Face ROI", 0.0

    # Calculate blur using Laplacian variance
    laplacian_var = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())

    if laplacian_var < 35.0:
        return False, "Blurry Image / Hold Still", laplacian_var

    quality_label = "Excellent Quality" if laplacian_var > 120 else "Good Quality"
    return True, quality_label, round(laplacian_var, 1)


def save_face_sample(gray_img, face_rect, save_dir: str, count: int, target_size=(200, 200)) -> str:
    x, y, w, h = face_rect
    margin_w = int(w * 0.1)
    margin_h = int(h * 0.1)
    img_h, img_w = gray_img.shape[:2]

    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_w, x + w + margin_w)
    y2 = min(img_h, y + h + margin_h)

    face_roi = gray_img[y1:y2, x1:x2]
    if face_roi.size == 0:
        face_roi = gray_img[y:y+h, x:x+w]

    resized_face = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_CUBIC)
    file_path = os.path.join(save_dir, f"sample_{count:03d}.jpg")
    cv2.imwrite(file_path, resized_face)
    return file_path


POSE_INSTRUCTIONS = [
    "Look directly at the camera",
    "Slowly tilt your head slightly left",
    "Slowly tilt your head slightly right",
    "Tilt your chin slightly up",
    "Tilt your chin slightly down",
    "Smile naturally or change facial expression",
]


def get_pose_instruction(sample_count: int) -> str:
    idx = (sample_count // 5) % len(POSE_INSTRUCTIONS)
    return POSE_INSTRUCTIONS[idx]
