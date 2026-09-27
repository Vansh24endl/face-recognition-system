import os
import json
import cv2
import numpy as np
from utils.face_detection import FaceDetector
from utils.expression_analyzer import ExpressionAnalyzer


def get_trainer_paths() -> tuple[str, str]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trainer_dir = os.path.join(project_root, "trainer")
    yml_path = os.path.join(trainer_dir, "trainer.yml")
    json_path = os.path.join(trainer_dir, "labels.json")
    return yml_path, json_path


def is_model_trained() -> bool:
    yml_path, json_path = get_trainer_paths()
    return os.path.exists(yml_path) and os.path.exists(json_path) and os.path.getsize(yml_path) > 0


class RealTimeRecognizer:
    def __init__(self, confidence_threshold: float = 75.0):
        self.yml_path, self.json_path = get_trainer_paths()
        self.confidence_threshold = confidence_threshold
        self.recognizer = None
        self.labels_map = {}
        self.is_loaded = False
        self.expression_analyzer = ExpressionAnalyzer()
        self.frame_counter = 0
        self.cached_expressions = {}  # Cache expressions to avoid lagging video
        self.load_model()

    def load_model(self) -> bool:
        if not is_model_trained():
            self.is_loaded = False
            return False

        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(self.yml_path)

            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.labels_map = json.load(f)

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading face recognizer model: {e}")
            self.is_loaded = False
            return False

    def predict_face(self, gray_face) -> dict:
        if not self.is_loaded or self.recognizer is None:
            return {
                "name": "Unknown",
                "person_id": "N/A",
                "confidence": 0,
                "distance": 999.0,
                "is_known": False
            }

        resized_face = cv2.resize(gray_face, (200, 200), interpolation=cv2.INTER_CUBIC)
        label, distance = self.recognizer.predict(resized_face)

        confidence_pct = max(0, min(100, int(100 - distance)))
        str_label = str(label)

        if distance <= self.confidence_threshold and str_label in self.labels_map:
            info = self.labels_map[str_label]
            return {
                "name": info["name"],
                "person_id": info["person_id"],
                "confidence": confidence_pct,
                "distance": round(distance, 2),
                "is_known": True
            }
        else:
            return {
                "name": "Unknown",
                "person_id": "N/A",
                "confidence": confidence_pct,
                "distance": round(distance, 2),
                "is_known": False
            }

    def process_frame(self, frame, detector: FaceDetector) -> tuple[np.ndarray, list[dict]]:
        """Processes video frame for both face recognition and facial expression analysis.
        Returns (annotated_frame, list_of_combined_results).
        """
        self.frame_counter += 1
        annotated_frame = frame.copy()
        faces, gray = detector.detect(frame)
        results = []

        for idx, (x, y, w, h) in enumerate(faces):
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_bgr = frame[y:y+h, x:x+w]

            if face_roi_gray.size == 0:
                continue

            # 1. Face Recognition (LBPH)
            rec_result = self.predict_face(face_roi_gray)

            # 2. Facial Expression Analysis (Cached every 3 frames for zero lag)
            cache_key = f"face_{idx}"
            if self.frame_counter % 3 == 0 or cache_key not in self.cached_expressions:
                exp_result = self.expression_analyzer.analyze_face(face_roi_bgr)
                self.cached_expressions[cache_key] = exp_result
            else:
                exp_result = self.cached_expressions[cache_key]

            combined = {
                "name": rec_result["name"],
                "person_id": rec_result["person_id"],
                "recognition_confidence": rec_result["confidence"],
                "distance": rec_result["distance"],
                "is_known": rec_result["is_known"],
                "expression": exp_result["dominant_expression"],
                "expression_emoji": exp_result["emoji"],
                "expression_confidence": exp_result["confidence"],
                "expression_probabilities": exp_result["all_probabilities"],
                "bbox": (x, y, w, h)
            }
            results.append(combined)

            # Render Bounding Box and Overlay Information
            box_color = (0, 255, 0) if rec_result["is_known"] else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)

            # Lines to display on banner
            name_str = f"Name: {rec_result['name']}"
            rec_str = f"Rec: {rec_result['confidence']}%" if rec_result['is_known'] else "Rec: Unknown"
            exp_str = f"Exp: {exp_result['emoji']} {exp_result['dominant_expression']} ({exp_result['confidence']}%)"

            # Draw background overlay box above bounding box
            line_height = 18
            banner_h = 3 * line_height + 10
            banner_w = max(180, w)

            banner_y1 = max(0, y - banner_h)
            banner_y2 = y

            cv2.rectangle(
                annotated_frame,
                (x, banner_y1),
                (x + banner_w, banner_y2),
                (15, 20, 28),
                -1
            )
            cv2.rectangle(
                annotated_frame,
                (x, banner_y1),
                (x + banner_w, banner_y2),
                box_color,
                1
            )

            # Draw Text Lines
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            font_thick = 1

            cv2.putText(annotated_frame, name_str, (x + 6, banner_y1 + 16), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
            cv2.putText(annotated_frame, rec_str, (x + 6, banner_y1 + 32), font, font_scale, (200, 230, 255), font_thick, cv2.LINE_AA)
            cv2.putText(annotated_frame, exp_str, (x + 6, banner_y1 + 48), font, font_scale, (100, 255, 200), font_thick, cv2.LINE_AA)

        return annotated_frame, results
