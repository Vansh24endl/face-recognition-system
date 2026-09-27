import os
import time
import json
import shutil
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime

# Import custom modular utilities
from utils.face_detection import FaceDetector, get_cascade_path
from utils.face_capture import (
    get_dataset_dir,
    sanitize_name,
    sanitize_id,
    check_existing_id,
    create_person_directory,
    test_working_camera,
    save_face_sample,
    evaluate_face_quality,
    get_pose_instruction
)
from utils.trainer import train_model, get_trainer_dir
from utils.recognizer import RealTimeRecognizer, is_model_trained, get_trainer_paths
from utils.expression_analyzer import ExpressionAnalyzer, EMOTIONS, EMOJI_MAP

# Streamlit Page Config
st.set_page_config(
    page_title="AI Face Intelligence System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Modern AI Dashboard Dark CSS
CUSTOM_CSS = """
<style>
    /* Dark AI Theme Variables */
    :root {
        --bg-color: #0b0f19;
        --card-bg: #131b2e;
        --card-border: #1e293b;
        --primary-accent: #6366f1;
        --secondary-accent: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-primary);
    }
    
    /* Title Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 26px 36px;
        border-radius: 16px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 24px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .header-banner h1 {
        color: #ffffff !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        margin: 0 !important;
    }
    .header-banner p {
        color: #a5b4fc !important;
        font-size: 1.1rem !important;
        margin-top: 6px !important;
        margin-bottom: 0 !important;
    }

    /* Top Metric Stat Cards */
    .stat-card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: var(--primary-accent);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.2);
    }
    .stat-title {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .stat-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
    }
    .stat-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .badge-purple { background-color: rgba(99, 102, 241, 0.2); color: #818cf8; }
    .badge-cyan { background-color: rgba(6, 182, 212, 0.2); color: #22d3ee; }
    .badge-green { background-color: rgba(16, 185, 129, 0.2); color: #34d399; }
    .badge-amber { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; }

    /* Overview Section Cards */
    .overview-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .overview-card h4 {
        color: #6366f1;
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 1.1rem;
    }

    /* Privacy Banner */
    .privacy-notice {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 12px 20px;
        color: #93c5fd;
        font-size: 0.9rem;
        margin-top: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #1e293b;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_dataset_stats():
    dataset_dir = get_dataset_dir()
    if not os.path.exists(dataset_dir):
        return 0, 0, []

    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    total_people = len(folders)
    total_images = 0
    people_details = []

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        img_count = len(images)
        total_images += img_count

        parts = folder.rsplit('_', 1)
        if len(parts) == 2:
            p_name, p_id = parts[0].replace('_', ' '), parts[1]
        else:
            p_name, p_id = folder, folder

        people_details.append({
            "folder": folder,
            "name": p_name,
            "id": p_id,
            "image_count": img_count,
            "path": folder_path
        })

    return total_people, total_images, people_details


# Initialize session state for expression history log
if "expression_history" not in st.session_state:
    st.session_state.expression_history = []

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=65)
st.sidebar.title("AI FACE INTELLIGENCE")

navigation_choice = st.sidebar.radio(
    "Modules Navigation:",
    [
        "📊 Dashboard",
        "👤 Register Face",
        "⚙️ Train Model",
        "🔍 Live Recognition",
        "🎭 Expression Analysis",
        "👥 Registered People",
        "ℹ️ About System"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Health")
total_people, total_images, people_list = get_dataset_stats()
model_status = is_model_trained()
working_cam = test_working_camera()

st.sidebar.write(f"👥 **Registered:** `{total_people}`")
st.sidebar.write(f"🖼️ **Images:** `{total_images}`")

if model_status:
    st.sidebar.success("LBPH Model: READY")
else:
    st.sidebar.warning("LBPH Model: NOT TRAINED")

if working_cam != -1:
    st.sidebar.success(f"Camera: Index {working_cam}")
else:
    st.sidebar.error("Camera: Disconnected")


# ==========================================
# PART 1 — REDESIGN DASHBOARD
# ==========================================
if navigation_choice == "📊 Dashboard":
    st.markdown("""
        <div class="header-banner">
            <h1>AI Face Intelligence System</h1>
            <p>Real-time Face Recognition & Facial Expression Analysis</p>
        </div>
    """, unsafe_allow_html=True)

    # Top Statistics Cards (5 Cards)
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">Registered People</div>
                <div class="stat-value">{total_people}</div>
                <div class="stat-badge badge-purple">Active Enrolled</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">Training Images</div>
                <div class="stat-value">{total_images}</div>
                <div class="stat-badge badge-cyan">Face Samples</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        status_str = "READY" if model_status else "NOT TRAINED"
        badge_c = "badge-green" if model_status else "badge-amber"
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">Recognition Model</div>
                <div class="stat-value" style="font-size: 1.5rem; margin-top: 6px;">{status_str}</div>
                <div class="stat-badge {badge_c}">LBPH Classifier</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        cam_str = f"ONLINE ({working_cam})" if working_cam != -1 else "OFFLINE"
        cam_b = "badge-green" if working_cam != -1 else "badge-amber"
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">Camera Status</div>
                <div class="stat-value" style="font-size: 1.5rem; margin-top: 6px;">{cam_str}</div>
                <div class="stat-badge {cam_b}">Hardware Video</div>
            </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Expressions</div>
                <div class="stat-value">7</div>
                <div class="stat-badge badge-purple">AI FER Classes</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### ⚙️ System Overview")
    ov1, ov2, ov3 = st.columns(3)

    with ov1:
        st.markdown(f"""
            <div class="overview-card">
                <h4>👤 Face Recognition</h4>
                <p>● <b>Model:</b> Local Binary Patterns Histogram (LBPH)</p>
                <p>● <b>Status:</b> {'Ready' if model_status else 'Needs Training'}</p>
                <p>● <b>Registered Faces:</b> {total_people}</p>
            </div>
        """, unsafe_allow_html=True)

    with ov2:
        st.markdown("""
            <div class="overview-card">
                <h4>🎭 Expression Analysis</h4>
                <p>● <b>Model:</b> AI Expression Model (ONNX FERPlus / DeepFace)</p>
                <p>● <b>Status:</b> Ready</p>
                <p>● <b>Supported Expressions:</b> 7 Classes (Happy, Sad, Neutral, Angry, Surprise, Fear, Disgust)</p>
            </div>
        """, unsafe_allow_html=True)

    with ov3:
        st.markdown(f"""
            <div class="overview-card">
                <h4>📹 Camera & Hardware</h4>
                <p>● <b>Status:</b> {'Connected' if working_cam != -1 else 'Disconnected'}</p>
                <p>● <b>Camera Index:</b> {working_cam if working_cam != -1 else 'None'}</p>
                <p>● <b>Face Detector:</b> Haar Cascade Classifier</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="privacy-notice">
            🔒 <b>Privacy Notice:</b> All face images and biometric features are processed locally on this device.
            The application does not upload facial data or expression history to external servers.
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# PART 2 & PART 3 — IMPROVED FACE REGISTRATION
# ==========================================
elif navigation_choice == "👤 Register Face":
    st.markdown("""
        <div class="header-banner">
            <h1>Register New Person</h1>
            <p>Smart multi-angle face sample acquisition with real-time quality evaluation</p>
        </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.3])

    with col_left:
        st.subheader("Step 1: Person Information")
        person_name = st.text_input("Full Name", placeholder="e.g. Vansh Sharma")
        person_id = st.text_input("Person ID / Roll No", placeholder="e.g. 101")
        sample_target = st.number_input("Samples Target", min_value=10, max_value=100, value=30, step=5)

        capture_mode = st.radio("Capture Method:", ["📷 Live Webcam Stream (Auto Capture)", "📸 Browser Snapshot (Manual)"])

        st.markdown("---")
        start_registration = st.button("🚀 Start Registration Pipeline", type="primary")

    with col_right:
        st.subheader("Step 2 & 3: Camera Preview & Smart Capture")
        cam_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        info_col1, info_col2, info_col3 = st.columns(3)
        stat_status = info_col1.empty()
        stat_count = info_col2.empty()
        stat_quality = info_col3.empty()

        instruction_box = st.empty()

        if start_registration:
            if not person_name.strip():
                st.error("Please enter a valid Person Name.")
            elif not person_id.strip():
                st.error("Please enter a valid Person ID.")
            else:
                is_existing, exist_folder = check_existing_id(person_id)
                if is_existing:
                    st.error(f"Person ID `{person_id}` already exists in database ({exist_folder}). Please use a unique ID.")
                else:
                    target_dir = create_person_directory(person_name, person_id)

                    if capture_mode == "📷 Live Webcam Stream (Auto Capture)":
                        cam_idx = test_working_camera()
                        if cam_idx == -1:
                            st.error("No working camera detected on this system!")
                        else:
                            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                            detector = FaceDetector()
                            count = 0

                            while cap.isOpened() and count < sample_target:
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    st.error("Error reading camera frame.")
                                    break

                                faces, gray = detector.detect(frame)
                                annotated_frame = frame.copy()

                                # Pose Guidance Instruction
                                current_pose = get_pose_instruction(count)
                                instruction_box.info(f"💡 **Instruction:** {current_pose}")

                                if len(faces) == 1:
                                    (x, y, w, h) = faces[0]
                                    is_valid_quality, qual_msg, qual_score = evaluate_face_quality(gray, (x, y, w, h))

                                    stat_status.markdown("Face Status:\n🟢 **Face Detected**")
                                    stat_quality.markdown(f"Quality:\n**{qual_msg}**")

                                    if is_valid_quality:
                                        count += 1
                                        save_face_sample(gray, (x, y, w, h), target_dir, count)

                                        # Draw Bounding Box & Progress
                                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                        cv2.putText(
                                            annotated_frame,
                                            f"Captured: {count}/{sample_target}",
                                            (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.65,
                                            (0, 255, 0),
                                            2
                                        )

                                        progress_bar.progress(count / sample_target)
                                        stat_count.markdown(f"Capture Progress:\n**{count} / {sample_target}**")
                                    else:
                                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                                        cv2.putText(
                                            annotated_frame,
                                            qual_msg,
                                            (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.65,
                                            (0, 165, 255),
                                            2
                                        )

                                elif len(faces) > 1:
                                    stat_status.markdown("Face Status:\n⚠️ **Multiple Faces**")
                                    stat_quality.markdown("Quality:\n**Only 1 Face Allowed**")
                                    cv2.putText(
                                        annotated_frame,
                                        "Only one face should be visible!",
                                        (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 0, 255),
                                        2
                                    )
                                else:
                                    stat_status.markdown("Face Status:\n🔴 **No Face**")
                                    stat_quality.markdown("Quality:\n**Keep Face Visible**")
                                    cv2.putText(
                                        annotated_frame,
                                        "No face detected!",
                                        (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 0, 255),
                                        2
                                    )

                                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                cam_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                time.sleep(0.04)

                            cap.release()

                            if count >= sample_target:
                                st.balloons()
                                st.success(f"🎉 **{sample_target} / {sample_target} images captured successfully** for {person_name}!")
                                
                                # Step 4: Display Captured Faces Gallery
                                st.subheader("Step 4: Captured Dataset Gallery")
                                sample_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png'))][:15]
                                cols = st.columns(5)
                                for idx, f_name in enumerate(sample_files):
                                    img_p = os.path.join(target_dir, f_name)
                                    cols[idx % 5].image(img_p, caption=f"Sample {idx+1}", width=110)

        if capture_mode == "📸 Browser Snapshot (Manual)" and start_registration:
            img_file_buffer = st.camera_input("Take a face snapshot")
            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                detector = FaceDetector()
                faces, gray = detector.detect(cv2_img)

                if len(faces) > 0:
                    target_dir = create_person_directory(person_name, person_id)
                    (x, y, w, h) = faces[0]
                    for idx in range(1, sample_target + 1):
                        save_face_sample(gray, (x, y, w, h), target_dir, idx)
                    st.success(f"Saved {sample_target} snapshot samples for {person_name}!")
                else:
                    st.error("No face detected in snapshot buffer. Please position face clearly.")


# ==========================================
# PART 3 (TRAIN MODEL)
# ==========================================
elif navigation_choice == "⚙️ Train Model":
    st.markdown("""
        <div class="header-banner">
            <h1>Train Face Recognizer Model</h1>
            <p>Train OpenCV LBPH Face Recognizer on captured face datasets</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📊 Dataset Overview")
        st.write(f"- **Registered Individuals:** `{total_people}`")
        st.write(f"- **Total Face Samples:** `{total_images}`")

        if people_list:
            st.markdown("#### Enrolled Profiles:")
            for p in people_list:
                st.write(f"- **{p['name']}** (ID: `{p['id']}`) — `{p['image_count']}` samples")

        st.markdown("---")
        start_train = st.button("🚀 Train LBPH Classifier", type="primary")

    with col2:
        st.subheader("💡 Model Specifications")
        st.info("""
        **Classifier:** Local Binary Patterns Histogram (LBPH)
        
        **Pipeline Execution:**
        1. Reads grayscale face ROI images from `dataset/`.
        2. Assigns numeric class index per person.
        3. Computes LBP codes over $8 \\times 8$ local cell histograms.
        4. Saves binary weights to `trainer/trainer.yml`.
        5. Writes metadata index to `trainer/labels.json`.
        """)

    if start_train:
        if total_people == 0 or total_images == 0:
            st.error("Cannot train model! Dataset is empty. Register at least one face first.")
        else:
            with st.spinner("Training LBPH model..."):
                progress = st.progress(0)
                for i in range(1, 85):
                    time.sleep(0.01)
                    progress.progress(i)

                results = train_model()
                progress.progress(100)

                if results["success"]:
                    st.balloons()
                    st.success(f"✅ {results['message']}")
                    st.json({
                        "Trained People Count": results["num_people"],
                        "Trained Images Count": results["num_images"],
                        "Weights Saved At": results["model_path"],
                        "Labels Index Saved At": results["labels_path"]
                    })
                else:
                    st.error(f"❌ Training Failed: {results['message']}")


# ==========================================
# PART 4, 5, 6, 7 — LIVE RECOGNITION & EXPRESSION INTEGRATION
# ==========================================
elif navigation_choice == "🔍 Live Recognition":
    st.markdown("""
        <div class="header-banner">
            <h1>Live Face Intelligence & Expression Stream</h1>
            <p>Real-time dual pipeline: LBPH Face Recognition + AI Facial Expression Analysis</p>
        </div>
    """, unsafe_allow_html=True)

    if not model_status:
        st.error("⚠️ LBPH Face Recognizer Model is not trained yet!")
        st.info("Go to **Train Model** module to train the model on dataset images first.")
    else:
        col_controls, col_feed = st.columns([1, 2])

        with col_controls:
            st.subheader("⚙️ Stream Configuration")
            cam_idx = st.selectbox("Select Camera Device", [working_cam if working_cam != -1 else 0, 1, 2])
            conf_thresh = st.slider(
                "Recognition Distance Threshold",
                min_value=30,
                max_value=120,
                value=75,
                help="Lower value requires stricter histogram match."
            )

            st.markdown("---")
            run_stream = st.checkbox("▶️ Start Live Camera Stream", value=False)

        with col_feed:
            st.subheader("📹 Real-Time AI Camera Feed")
            frame_container = st.empty()
            exp_panel = st.empty()
            hist_container = st.empty()

            if run_stream:
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                detector = FaceDetector()
                recognizer = RealTimeRecognizer(confidence_threshold=conf_thresh)

                if not cap.isOpened():
                    st.error(f"Unable to access camera index {cam_idx}.")
                else:
                    while run_stream:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Failed to capture video frame.")
                            break

                        annotated_frame, detections = recognizer.process_frame(frame, detector)

                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_container.image(rgb_frame, channels="RGB", use_container_width=True)

                        # Render Expression Panel & Append History
                        if detections:
                            primary_det = detections[0]
                            dom_exp = primary_det["expression"]
                            dom_emoji = primary_det["expression_emoji"]
                            dom_conf = primary_det["expression_confidence"]
                            all_probs = primary_det["expression_probabilities"]

                            # Add to temporary session history
                            curr_time = datetime.now().strftime("%H:%M:%S")
                            st.session_state.expression_history.insert(0, {
                                "Time": curr_time,
                                "Person": primary_det["name"],
                                "Recognition Conf": f"{primary_det['recognition_confidence']}%" if primary_det["is_known"] else "Unknown",
                                "Expression": f"{dom_emoji} {dom_exp}",
                                "Exp Confidence": f"{dom_conf}%"
                            })
                            # Keep last 10 entries
                            st.session_state.expression_history = st.session_state.expression_history[:10]

                            # Display Real-Time Expression Panel (Part 6)
                            panel_html = f"### 🎭 FACIAL EXPRESSION ANALYSIS\n"
                            panel_html += f"**Current Predicted Expression:** {dom_emoji} **{dom_exp}** (`{dom_conf}% confidence`)\n\n"
                            exp_panel.markdown(panel_html)

                            # Display Proportions Bar
                            for exp_name, prob_val in all_probs.items():
                                exp_panel.progress(int(prob_val) / 100, text=f"{EMOJI_MAP.get(exp_name, '')} {exp_name}: {prob_val}%")

                        time.sleep(0.03)

                    cap.release()

        # PART 7: Expression History Table
        if st.session_state.expression_history:
            st.markdown("### 📜 Session Expression History Log")
            st.table(st.session_state.expression_history)


# ==========================================
# PART 5 (EXPRESSION ANALYSIS LAB)
# ==========================================
elif navigation_choice == "🎭 Expression Analysis":
    st.markdown("""
        <div class="header-banner">
            <h1>Facial Expression Analysis Lab</h1>
            <p>Analyze emotion distributions using deep convolutional neural network features</p>
        </div>
    """, unsafe_allow_html=True)

    analyzer = ExpressionAnalyzer()
    st.write(f"**Engine Status:** `{'Ready' if analyzer.is_ready else 'Not Loaded'}` | **Model:** `{analyzer.model_name}`")

    tab_upload, tab_cam = st.tabs(["📁 Upload Image", "📸 Live Snapshot"])

    with tab_upload:
        uploaded_file = st.file_uploader("Choose a face image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image.convert('RGB'))
            bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            detector = FaceDetector()
            faces, _ = detector.detect(bgr_img)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_crop = bgr_img[y:y+h, x:x+w]
                exp_res = analyzer.analyze_face(face_crop)

                col_img, col_metrics = st.columns(2)
                with col_img:
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                with col_metrics:
                    st.markdown(f"### Predicted Facial Expression:\n## {exp_res['emoji']} {exp_res['dominant_expression']}")
                    st.write(f"**Confidence:** `{exp_res['confidence']}%`")

                    st.markdown("#### Probability Breakdown:")
                    for e_name, p_val in exp_res['all_probabilities'].items():
                        st.progress(int(p_val) / 100, text=f"{EMOJI_MAP.get(e_name, '')} {e_name}: {p_val}%")
            else:
                st.error("No face detected in uploaded image.")

    with tab_cam:
        snap_file = st.camera_input("Take photo for expression testing")
        if snap_file is not None:
            bytes_d = snap_file.getvalue()
            bgr_img = cv2.imdecode(np.frombuffer(bytes_d, np.uint8), cv2.IMREAD_COLOR)
            detector = FaceDetector()
            faces, _ = detector.detect(bgr_img)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_crop = bgr_img[y:y+h, x:x+w]
                exp_res = analyzer.analyze_face(face_crop)

                st.markdown(f"### Predicted Expression: {exp_res['emoji']} **{exp_res['dominant_expression']}** ({exp_res['confidence']}%)")
                for e_name, p_val in exp_res['all_probabilities'].items():
                    st.progress(int(p_val) / 100, text=f"{EMOJI_MAP.get(e_name, '')} {e_name}: {p_val}%")


# ==========================================
# PART 9 — REGISTERED PEOPLE PAGE
# ==========================================
elif navigation_choice == "👥 Registered People":
    st.markdown("""
        <div class="header-banner">
            <h1>Registered People Directory</h1>
            <p>Audit and manage enrolled person dataset folders</p>
        </div>
    """, unsafe_allow_html=True)

    if total_people == 0:
        st.info("No registered individuals found in dataset directory.")
    else:
        st.write(f"Total Enrolled Persons: `{total_people}`")

        # Summary Table
        table_data = []
        for p in people_list:
            table_data.append({
                "ID": p["id"],
                "Name": p["name"],
                "Images": p["image_count"],
                "Folder": p["folder"]
            })
        st.table(table_data)

        st.markdown("---")
        st.subheader("🗑️ Delete Enrolled Person")

        del_person = st.selectbox("Select Person to Delete", [f"{p['name']} (ID: {p['id']})" for p in people_list])
        if st.button("Confirm Delete", type="primary"):
            sel_id = del_person.split("(ID: ")[1].rstrip(")")
            for p in people_list:
                if p["id"] == sel_id:
                    try:
                        shutil.rmtree(p["path"])
                        st.success(f"Deleted dataset folder for {p['name']}.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting folder: {e}")


# ==========================================
# PART 14 — ABOUT PAGE
# ==========================================
elif navigation_choice == "ℹ️ About System":
    st.markdown("""
        <div class="header-banner">
            <h1>About AI Face Intelligence System</h1>
            <p>Comprehensive Documentation & System Architecture</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 📌 1. Project Objective
    To build an integrated, offline **AI Face Intelligence System** capable of real-time face detection, dataset sample collection, LBPH face identification, and AI-driven facial expression analysis.

    ---

    ### 🔍 2. Face Detection (Haar Cascade)
    Operates via Paul Viola and Michael Jones' algorithm utilizing rectangular Haar-like features, Integral Images for rapid computation, AdaBoost feature selection, and a Cascade of Classifiers.

    ---

    ### 🧬 3. Face Recognition (LBPH)
    Employs Local Binary Patterns Histograms. Divides cropped face ROI into $8 \\times 8$ cells, extracts pixel neighborhood binary patterns, and constructs spatial feature vector histograms. Calculates similarity using Chi-Square distance.

    ---

    ### 🎭 4. Facial Expression Analysis
    Uses a deep Convolutional Neural Network (ONNX FERPlus / DeepFace) running locally in OpenCV DNN. Evaluates probabilities across 7 expression categories: **Neutral, Happy, Surprise, Sad, Angry, Fear, Disgust**.

    ---

    ### 🛠️ 5. Technology Stack
    - **Language:** Python 3.10+
    - **Vision Core:** OpenCV (`opencv-contrib-python`)
    - **Detection:** Haar Cascade (`haarcascade_frontalface_default.xml`)
    - **Recognition:** LBPH Face Recognizer
    - **Expression Model:** OpenCV DNN ONNX FERPlus / DeepFace
    - **UI Framework:** Streamlit Dark Custom Theme
    - **Math / Arrays:** NumPy

    ---

    ### 🔒 6. Privacy & Security Statement
    All biometric data, face samples, model weights, and expression history are processed and persisted **100% locally** on your device. No facial embeddings or images are uploaded to third-party cloud servers.
    """)

# Footer
st.markdown("""
    <div class="footer">
        AI Face Intelligence System | Real-time Recognition & Facial Expression Analysis
    </div>
""", unsafe_allow_html=True)
