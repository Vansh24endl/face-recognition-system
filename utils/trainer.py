import os
import json
import cv2
import numpy as np
from utils.face_detection import FaceDetector, get_cascade_path


def get_trainer_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trainer_dir = os.path.join(project_root, "trainer")
    os.makedirs(trainer_dir, exist_ok=True)
    return trainer_dir


def train_model() -> dict:
    """Reads dataset, trains LBPH Face Recognizer, and saves trainer.yml and labels.json.
    Returns dictionary with training metrics and status.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "dataset")
    trainer_dir = get_trainer_dir()

    trainer_yml_path = os.path.join(trainer_dir, "trainer.yml")
    labels_json_path = os.path.join(trainer_dir, "labels.json")

    if not os.path.exists(dataset_dir):
        return {
            "success": False,
            "message": "Dataset directory does not exist.",
            "num_people": 0,
            "num_images": 0
        }

    person_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    if not person_folders:
        return {
            "success": False,
            "message": "No registered face folders found in dataset/",
            "num_people": 0,
            "num_images": 0
        }

    # Initialize LBPH Recognizer & Face Detector
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
    except AttributeError:
        return {
            "success": False,
            "message": "OpenCV LBPH module not found. Make sure opencv-contrib-python is installed.",
            "num_people": 0,
            "num_images": 0
        }

    detector = FaceDetector()

    faces = []
    labels = []
    label_map = {}
    label_counter = 0
    total_images_processed = 0

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for folder in sorted(person_folders):
        folder_path = os.path.join(dataset_dir, folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

        if not image_files:
            continue

        # Extract name and id from folder name (Format: Name_ID)
        parts = folder.rsplit('_', 1)
        if len(parts) == 2:
            person_name = parts[0].replace('_', ' ')
            person_id = parts[1]
        else:
            person_name = folder
            person_id = folder

        numeric_label = label_counter
        label_map[str(numeric_label)] = {
            "name": person_name,
            "person_id": person_id,
            "folder": folder
        }
        label_counter += 1

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # Standardize size if not cropped
            if img.shape[0] != 200 or img.shape[1] != 200:
                # Detect face or resize directly
                detected_faces, _ = detector.detect(img)
                if len(detected_faces) > 0:
                    x, y, w, h = detected_faces[0]
                    face_crop = img[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_crop, (200, 200))
                else:
                    face_roi = cv2.resize(img, (200, 200))
            else:
                face_roi = img

            faces.append(face_roi)
            labels.append(numeric_label)
            total_images_processed += 1

    if not faces:
        return {
            "success": False,
            "message": "No valid face images found across dataset folders.",
            "num_people": 0,
            "num_images": 0
        }

    # Train model
    recognizer.train(faces, np.array(labels, dtype=np.int32))

    # Save trained model
    recognizer.write(trainer_yml_path)

    # Save labels mapping
    with open(labels_json_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=4)

    return {
        "success": True,
        "message": f"Successfully trained model with {len(label_map)} people and {total_images_processed} images.",
        "num_people": len(label_map),
        "num_images": total_images_processed,
        "model_path": trainer_yml_path,
        "labels_path": labels_json_path
    }
