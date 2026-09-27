# COLLEGE PRACTICAL LAB REPORT

## 1. TITLE
**Face Recognition System using Python and OpenCV**

---

## 2. AIM
To design, implement, and evaluate a computer vision-based real-time **Face Recognition System** capable of face detection, dataset registration, feature extraction, model training, and real-time facial identification using Python, OpenCV, and Streamlit.

---

## 3. OBJECTIVES
1. To understand the fundamental concepts of digital image processing, feature extraction, and pattern recognition.
2. To implement face detection using Paul Viola and Michael Jones' **Haar Cascade Classifier**.
3. To implement face identification using Ahonen et al.'s **Local Binary Patterns Histograms (LBPH)** algorithm.
4. To develop an interactive, offline desktop dashboard using **Streamlit** for dataset management and real-time inference.
5. To evaluate system performance under varying threshold conditions and analyze confidence scores.

---

## 4. SOFTWARE REQUIREMENTS
- **Operating System:** Windows 10 / 11 (64-bit)
- **Programming Language:** Python 3.10 or higher
- **Computer Vision Library:** OpenCV Contrib (`opencv-contrib-python >= 4.8.0`)
- **Numerical Library:** NumPy (`>= 1.24.0`)
- **Web Application Framework:** Streamlit (`>= 1.28.0`)
- **Image Processing Library:** Pillow (`>= 9.5.0`)
- **IDE / Code Editor:** VS Code / PyCharm / AntiGravity IDE

---

## 5. HARDWARE REQUIREMENTS
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher (Quad-Core recommended)
- **RAM:** 4 GB minimum (8 GB recommended)
- **Storage:** 500 MB free space
- **Camera:** Standard Built-in HD Webcam or USB Web Camera (720p resolution minimum)

---

## 6. THEORY

### A. Face Detection vs. Face Recognition
- **Face Detection:** The process of locating and determining the spatial boundaries $(x, y, w, h)$ of human faces in an image or video frame. It answers the question: *"Is there a face in this image, and where is it?"*
- **Face Recognition:** The process of extracting unique facial features from a detected face region and comparing them against a database of known individuals to determine identity. It answers the question: *"Whose face is this?"*

### B. Haar Cascade Classifier (Viola-Jones Algorithm)
Haar Cascade is a machine learning object detection algorithm proposed by Paul Viola and Michael Jones in 2001. It operates through four major steps:
1. **Haar Features Selection:** Calculates scalar values by subtracting the sum of pixel intensities under dark rectangular regions from white rectangular regions.
   $$\text{Feature Value} = \sum \text{Pixels}_{\text{Dark}} - \sum \text{Pixels}_{\text{White}}$$
2. **Integral Image Representation:** Computes sub-rectangle pixel sums in constant time $O(1)$, drastically reducing computational overhead.
3. **AdaBoost Training:** Selects the most discriminating facial features from thousands of candidates and discards redundant features.
4. **Cascading Classifiers:** Chains weak classifiers into a multi-stage cascade. Sub-windows are evaluated sequentially; non-face regions are rejected immediately at early stages.

### C. Local Binary Patterns Histograms (LBPH)
LBPH is a powerful, computationally efficient texture descriptor introduced by Ojala et al. (1996) and adapted for face recognition by Ahonen et al. (2004).

1. **LBP Operator Computation:**
   For a pixel $c = (x_c, y_c)$ with intensity $P_c$ and $P$ neighbor pixels $p_n$ at radius $R$:
   $$LBP_{P, R}(x_c, y_c) = \sum_{n=0}^{P-1} s(P_n - P_c) \cdot 2^n$$
   where the threshold function $s(x)$ is defined as:
   $$s(x) = \begin{cases} 1 & \text{if } x \ge 0 \\ 0 & \text{if } x < 0 \end{cases}$$

2. **Grid Division & Histogram Concatenation:**
   - The face image is divided into $8 \times 8$ local spatial regions (cells).
   - A 256-bin histogram of LBP codes is computed for each cell.
   - All cell histograms are concatenated into a single spatial feature vector representing the face texture.

3. **Pattern Matching & Distance Metric:**
   The recognizer compares the feature vector of a query face $H_{query}$ against stored dataset histograms $H_{dataset}$ using Chi-Square distance ($\chi^2$) or Euclidean distance:
   $$D(H_1, H_2) = \sum_{i} \frac{(H_{1,i} - H_{2,i})^2}{H_{1,i} + H_{2,i}}$$
   - **Distance Interpretation:** Lower distance indicates higher similarity ($0 = \text{exact match}$). Distance value is mapped into a confidence percentage.

---

## 7. ALGORITHM

### Step 1: Face Registration Algorithm
1. Input `Person Name` and `Person ID`.
2. Validate inputs and verify that `Person ID` is unique in `dataset/`.
3. Open camera stream (`cv2.VideoCapture`).
4. For each frame:
   - Convert frame to grayscale and equalize histogram.
   - Detect face bounding box $(x, y, w, h)$ using Haar Cascade Classifier.
   - Crop face ROI, add padding, and resize to $200 \times 200$ pixels.
   - Save image to `dataset/PersonName_ID/sample_XXX.jpg`.
   - Increment sample count until target (30 images) is reached.
5. Release camera stream and display completion status.

### Step 2: Model Training Algorithm
1. Initialize `cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)`.
2. Read all image files from `dataset/*`.
3. Map each unique directory (`PersonName_ID`) to an integer label $0, 1, 2, \dots$
4. Load each image in grayscale, detect/verify face ROI, and append face matrix to `faces[]` and label to `labels[]`.
5. Execute `recognizer.train(faces, np.array(labels))`.
6. Save trained binary weights to `trainer/trainer.yml`.
7. Save label index dictionary to `trainer/labels.json`.

### Step 3: Real-Time Recognition Algorithm
1. Load `trainer/trainer.yml` and `trainer/labels.json`.
2. Open camera stream.
3. For each video frame:
   - Convert frame to grayscale.
   - Detect faces $(x, y, w, h)$ using Haar Cascade.
   - Crop face ROI and resize to $200 \times 200$.
   - Execute `label, distance = recognizer.predict(face_roi)`.
   - Calculate `confidence = max(0, 100 - distance)`.
   - If `distance <= threshold`:
     - Display Green bounding box, Person Name, ID, and Confidence %.
   - Else:
     - Display Red bounding box and "Unknown".
4. Render frame to UI.

---

## 8. STEP-BY-STEP PROCEDURE
1. Open Windows Command Prompt / Terminal and navigate to the project directory.
2. Run `setup.bat` to create virtual environment and install all packages from `requirements.txt`.
3. Execute `run.bat` to launch the Streamlit application.
4. On the Dashboard, verify that camera and directory dependencies are active.
5. Go to **Register Face**, fill in your details, and capture 30 face samples.
6. Go to **Train Model** and click **Start LBPH Model Training**.
7. Go to **Live Recognition**, enable the camera stream, and observe real-time identification.

---

## 9. IMPLEMENTATION
The implementation is organized into modular Python files:
- `app.py`: Streamlit frontend layout and interactive page routing.
- `utils/face_detection.py`: Wrapper for Haar Cascade detection and automatic XML resource fetch.
- `utils/face_capture.py`: Dataset folder creator and face sample extractor.
- `utils/trainer.py`: LBPH model trainer and persistence writer.
- `utils/recognizer.py`: Inference pipeline and bounding box renderer.

---

## 10. EXPECTED OUTPUT
1. **Registration:** Live webcam view highlighting detected face with green rectangle and progress counter (`Capturing: 15/30`).
2. **Training:** System logs showing total persons trained and `trainer/trainer.yml` generated.
3. **Live Recognition:** Green bounding box surrounding known face displaying `"Vansh (92%)"` or Red box displaying `"Unknown (Conf: 30%)"`.

---

## 11. APPLICATIONS
- **Educational Institutes:** Automated student attendance systems.
- **Corporate Offices:** Employee biometric entry authorization.
- **Security & Access Control:** Restricted area access gates.
- **Retail & Banking:** VIP customer recognition and fraud prevention.

---

## 12. ADVANTAGES
- **Fully Local & Offline:** No dependency on cloud APIs or internet connection.
- **Low Computational Cost:** Runs smoothly on standard CPU without dedicated GPU.
- **Monotonicity to Illumination:** LBPH pattern thresholding handles lighting changes better than raw pixel matching.
- **Incremental Scaling:** Easy to add new faces and retrain in seconds.

---

## 13. LIMITATIONS
- **Pose Sensitivity:** Works best for frontal faces (rotations $> 45^\circ$ degrade detection).
- **Occlusion Impact:** Heavy sunglasses, masks, or extreme darkness reduce confidence.

---

## 14. CONCLUSION
A robust, offline, real-time Face Recognition System was successfully developed using Python, OpenCV, and Streamlit. The system demonstrated accurate face detection via Haar Cascade and efficient classification using LBPH. The project meets all academic requirements for computer vision practicals.

---

## 15. VIVA VOCE QUESTIONS & DETAILED ANSWERS

### Q1: What is the main difference between Face Detection and Face Recognition?
**Answer:**
Face Detection locates human faces in an image and outputs their bounding box coordinates $(x, y, w, h)$. It does not identify who the person is. Face Recognition takes the detected face ROI, extracts facial feature vectors, and compares them against a known database to identify the specific individual.

### Q2: How does the Haar Cascade classifier work?
**Answer:**
Haar Cascade uses Haar-like rectangular features (edge, line, and four-rectangle features) to compute intensity differences across image regions. It uses an **Integral Image** to compute feature values in constant time $O(1)$, **AdaBoost** to select key features, and a **Cascade of Classifiers** to quickly discard non-face background regions.

### Q3: Why do we convert images to Grayscale before processing?
**Answer:**
Grayscale reduces image data complexity from 3 color channels (RGB) to 1 intensity channel (0–255), reducing memory and computation requirements by 66%. Object geometry and facial texture are preserved in grayscale.

### Q4: What is LBPH and why is it preferred over Eigenfaces/Fisherfaces for local applications?
**Answer:**
LBPH stands for **Local Binary Patterns Histograms**. Unlike Eigenfaces (PCA) or Fisherfaces (LDA) which look at global face images, LBPH analyzes local micro-structures of the face by dividing it into small cell grids. It is invariant to monotonic light changes, requires minimal training time, and allows instant offline training without complex matrix inversions.

### Q5: How is the Local Binary Pattern (LBP) code calculated for a pixel?
**Answer:**
For a center pixel, its intensity is compared with its 8 surrounding neighbors. If a neighbor's intensity is $\ge$ the center pixel's intensity, it is assigned a binary `1`; otherwise `0`. The resulting 8-bit binary number is converted into a decimal value (0 to 255) representing the local texture.

### Q6: How does LBPH represent a full face image?
**Answer:**
The face image is divided into spatial regions (e.g., $8 \times 8$ cells). An LBP histogram is constructed for each cell. All cell histograms are concatenated into a single high-dimensional feature vector.

### Q7: What does the confidence/distance value returned by `LBPHFaceRecognizer.predict()` mean?
**Answer:**
In OpenCV LBPH, `predict()` returns a `(label, distance)` tuple. The `distance` measures Chi-Square or Euclidean distance between histograms. **A lower distance value means a closer match (0 is an exact match)**. We map this distance into a percentage using $\text{Confidence \%} = \max(0, 100 - \text{distance})$.

### Q8: How does the system recognize an "Unknown" face?
**Answer:**
If the predicted distance exceeds a defined threshold (e.g., 75), the system considers the feature vector too far from any registered class in the dataset and labels the face as **"Unknown"**.

### Q9: Why is `opencv-contrib-python` required instead of standard `opencv-python`?
**Answer:**
The face recognition algorithms (`cv2.face.LBPHFaceRecognizer_create()`, `EigenFaceRecognizer`, `FisherFaceRecognizer`) are part of OpenCV's extra/contrib modules located in `opencv-contrib-python`. Standard `opencv-python` contains core vision tools but omits the `cv2.face` module.

### Q10: What is Histogram Equalization and why is it used?
**Answer:**
Histogram Equalization stretches the contrast range of an image by redistributing pixel intensity frequencies. It improves feature clarity in underexposed or poorly lit face images before detection/training.
