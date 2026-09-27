import os
import shutil
import urllib.request
import cv2


def get_cascade_path() -> str:
    """Ensures the Haar Cascade XML file exists in data/ directory and returns its absolute path."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    xml_name = "haarcascade_frontalface_default.xml"
    target_path = os.path.join(data_dir, xml_name)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        return target_path

    # Try copying from cv2.data
    try:
        cv2_cascade_path = os.path.join(cv2.data.haarcascades, xml_name)
        if os.path.exists(cv2_cascade_path):
            shutil.copy(cv2_cascade_path, target_path)
            return target_path
    except Exception:
        pass

    # Fallback: Download from official OpenCV repository
    url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml_name}"
    try:
        urllib.request.urlretrieve(url, target_path)
        return target_path
    except Exception as e:
        raise RuntimeError(f"Could not locate or download Haar Cascade classifier: {e}")


class FaceDetector:
    def __init__(self, scale_factor: float = 1.2, min_neighbors: int = 5, min_size: tuple = (30, 30)):
        cascade_path = get_cascade_path()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load Haar Cascade XML from {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image):
        """Detect faces in an image (BGR or Grayscale). Returns (faces, gray_img)."""
        if image is None:
            return [], None

        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Histogram Equalization for better contrast and accuracy
        gray_eq = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        return faces, gray

    def draw_faces(self, frame, faces, color=(0, 255, 0), thickness=2):
        """Draw bounding boxes around detected faces on frame."""
        frame_copy = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
        return frame_copy
