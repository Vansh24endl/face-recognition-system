import os
import urllib.request
import cv2
import numpy as np

# Emotion categories mapping for FERPlus model
EMOTIONS = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear']

EMOJI_MAP = {
    'Neutral': '😐',
    'Happy': '😊',
    'Surprise': '😲',
    'Sad': '😢',
    'Angry': '😡',
    'Disgust': '🤢',
    'Fear': '😨'
}


def get_onnx_model_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    onnx_path = os.path.join(models_dir, "emotion-ferplus.onnx")
    if os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 100000:
        return onnx_path

    # Download ONNX model from ONNX Model Zoo
    url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    try:
        urllib.request.urlretrieve(url, onnx_path)
        return onnx_path
    except Exception as e:
        print(f"Error downloading FERPlus ONNX model: {e}")
        return ""


class ExpressionAnalyzer:
    def __init__(self):
        self.net = None
        self.is_ready = False
        self.model_name = "ONNX-FERPlus (OpenCV DNN)"
        self.load_model()

    def load_model(self) -> bool:
        try:
            model_path = get_onnx_model_path()
            if model_path and os.path.exists(model_path):
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.is_ready = self.net is not None
                return self.is_ready
        except Exception as e:
            print(f"Failed to load OpenCV DNN Expression model: {e}")

        self.is_ready = False
        return False

    def analyze_face(self, face_img) -> dict:
        """Analyze face image (BGR or Grayscale ROI) and return emotion probabilities and top emotion.
        Returns dict with dominant_expression, emoji, confidence, and all_probabilities.
        """
        if not self.is_ready or self.net is None or face_img is None or face_img.size == 0:
            return {
                "dominant_expression": "Neutral",
                "emoji": "😐",
                "confidence": 50,
                "all_probabilities": {e: (50.0 if e == "Neutral" else 8.0) for e in EMOTIONS},
                "model_name": "Fallback Engine"
            }

        # Ensure grayscale
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_img.copy()

        # Resize to FERPlus 64x64 standard size
        resized = cv2.resize(gray_face, (64, 64), interpolation=cv2.INTER_AREA)

        # Create 4D Blob
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0,
            size=(64, 64),
            mean=0,
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)
        output = self.net.forward()

        # Raw scores out of FERPlus: 8 classes (0..7)
        raw_scores = output[0][:7]

        # Softmax normalization
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_scores / np.sum(exp_scores)

        # Map to percentage dict
        all_probs = {}
        for idx, emotion in enumerate(EMOTIONS):
            all_probs[emotion] = round(float(probabilities[idx] * 100), 1)

        # Top dominant emotion
        top_idx = int(np.argmax(probabilities))
        dominant_emotion = EMOTIONS[top_idx]
        confidence_pct = int(round(probabilities[top_idx] * 100))

        return {
            "dominant_expression": dominant_emotion,
            "emoji": EMOJI_MAP.get(dominant_emotion, "😐"),
            "confidence": confidence_pct,
            "all_probabilities": all_probs,
            "model_name": self.model_name
        }
