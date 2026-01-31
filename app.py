import streamlit as st
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import models
from torchvision.ops import nms
from PIL import Image
import tempfile
import subprocess
import os
import glob

from ocr.pipeline import (
    build_ocr_engine,
    build_orientation_detector,
    resize_if_needed,
    enhance_image,
    run_ocr_on_variants,
    run_with_orientation_check,
    rotate_image,
    draw_results as ocr_draw_results,
)

st.set_page_config(page_title="AI Pipeline — Detection & OCR", layout="wide")

# ── Model Loading (cached — loads once) ──────────────────────────────────────

@st.cache_resource
def load_detection_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
    detector.load_state_dict(torch.load("models/detector_final.pth", map_location=device))
    detector.to(device).eval()

    classifier = models.efficientnet_b3()
    classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 2)
    classifier.load_state_dict(torch.load("models/classifier_best.pth", map_location=device))
    classifier.to(device).eval()

    classify_tf = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return detector, classifier, classify_tf, device


@st.cache_resource
def load_ocr():
    return build_ocr_engine()


@st.cache_resource
def load_orientation_detector():
    return build_orientation_detector()


# ── Detection helpers ────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3


def apply_nms(boxes, scores, labels):
    if len(boxes) == 0:
        return boxes, scores, labels
    keep_indices = []
    for cid in labels.unique():
        mask = labels == cid
        keep = nms(boxes[mask], scores[mask], NMS_THRESHOLD)
        keep_indices.extend(torch.where(mask)[0][keep].tolist())
    idx = torch.tensor(keep_indices)
    return boxes[idx], scores[idx], labels[idx]


def process_frame(frame_rgb, detector, classifier, classify_tf, device):
    h, w = frame_rgb.shape[:2]
    img_tensor = T.ToTensor()(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        dets = detector(img_tensor)[0]

    boxes, scores, labels = dets["boxes"], dets["scores"], dets["labels"]
    keep = scores > CONF_THRESHOLD
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(boxes) > 0:
        boxes, scores, labels = apply_nms(boxes, scores, labels)

    results = []
    for box, det_score, _ in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(crop)
        crop_tensor = classify_tf(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = classifier(crop_tensor)
            probs = torch.softmax(pred, dim=1)[0]
            cls_idx = pred.argmax(1).item()
            cls_conf = probs[cls_idx].item()

        label_name = "ANIMAL" if cls_idx == 0 else "HUMAN"
        combined_conf = det_score.item() * cls_conf
        results.append(((x1, y1, x2, y2), label_name, combined_conf))

    return results


def draw_detections(frame_bgr, results):
    out = frame_bgr.copy()
    for (x1, y1, x2, y2), label, conf in results:
        color = (0, 255, 0) if label == "HUMAN" else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(out, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
        cv2.putText(out, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def reencode_for_browser(input_path):
    output_path = input_path.replace(".mp4", "_h264.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-loglevel", "error", output_path],
        check=True,
    )
    return output_path


def get_duration_options(total_seconds):
    options = []
    for s in [3, 5, 10, 20, 30]:
        if s < total_seconds:
            options.append(s)
    options.append(total_seconds)
    return options


# ── File discovery helpers ───────────────────────────────────────────────────

def list_sample_videos():
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join("test_videos", ext)))
    return sorted([os.path.basename(f) for f in files])


def list_sample_images():
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join("test_images", ext)))
    return sorted([os.path.basename(f) for f in files])


def list_detection_videos():
    """Find annotated detection videos in outputs/."""
    files = glob.glob(os.path.join("outputs", "*.mp4"))
    return sorted(files, key=os.path.getmtime, reverse=True)


def list_ocr_results():
    """Find OCR results that have both an annotated image and a JSON file."""
    jpg_files = glob.glob(os.path.join("outputs", "*_ocr.jpg"))
    pairs = []
    for jpg in sorted(jpg_files, key=os.path.getmtime, reverse=True):
        json_path = jpg.replace("_ocr.jpg", "_ocr.json")
        if os.path.exists(json_path):
            pairs.append((jpg, json_path))
    return pairs


def get_cached_h264_video(source_path):
    """Re-encode mp4v → H.264 once, cache in outputs/cached_h264/."""
    cache_dir = os.path.join("outputs", "cached_h264")
    os.makedirs(cache_dir, exist_ok=True)

    basename = os.path.basename(source_path)
    cached = os.path.join(cache_dir, basename)

    # Re-encode only if cache is missing or source is newer
    if not os.path.exists(cached) or os.path.getmtime(source_path) > os.path.getmtime(cached):
        subprocess.run(
            ["ffmpeg", "-y", "-i", source_path, "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-loglevel", "error", cached],
            check=True,
        )
    return cached


def format_file_size(path):
    """Human-readable file size."""
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ── TABS ─────────────────────────────────────────────────────────────────────

tab_detect, tab_ocr, tab_results = st.tabs(["Human & Animal Detection", "Industrial OCR", "Results Browser"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

with tab_detect:
    st.header("Human & Animal Detection")
    st.caption("Pick a sample video or upload your own. Select duration and run detection.")

    # --- Input selection: sample videos OR upload ---
    sample_videos = list_sample_videos()

    input_mode = st.radio(
        "Input source", ["Select from samples", "Upload a video"],
        horizontal=True, key="det_mode"
    )

    video_path = None  # will be set to a file path we can read with cv2

    if input_mode == "Select from samples" and sample_videos:
        chosen = st.selectbox("Sample videos", sample_videos, key="det_sample")
        video_path = os.path.join("test_videos", chosen)
    elif input_mode == "Upload a video":
        uploaded_video = st.file_uploader(
            "Upload video", type=["mp4", "avi", "mov", "mkv"], key="det_upload"
        )
        if uploaded_video is not None:
            tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_input.write(uploaded_video.read())
            tmp_input.flush()
            video_path = tmp_input.name
    elif input_mode == "Select from samples" and not sample_videos:
        st.info("No sample videos found in `test_videos/`. Upload one instead.")

    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_seconds = round(total_frames / fps, 1) if fps > 0 else 0
        cap.release()

        st.markdown(
            f"**Video info:** {width}x{height} &nbsp;|&nbsp; {fps:.0f} FPS "
            f"&nbsp;|&nbsp; {total_seconds}s &nbsp;|&nbsp; {total_frames} frames"
        )

        # Duration buttons
        options = get_duration_options(total_seconds)
        cols = st.columns(len(options))
        selected_seconds = None

        for i, sec in enumerate(options):
            label = f"Full ({sec:.0f}s)" if sec == total_seconds else f"{sec}s"
            if cols[i].button(label, key=f"dur_{sec}", width="stretch"):
                selected_seconds = sec

        if selected_seconds is not None:
            max_frames = int(fps * selected_seconds)

            detector, classifier, classify_tf, device = load_detection_models()

            tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_writer = cv2.VideoWriter(
                tmp_output.name, cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (width, height)
            )

            cap = cv2.VideoCapture(video_path)
            progress = st.progress(0, text="Processing frames...")
            frame_idx = 0
            human_total = 0
            animal_total = 0

            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = process_frame(rgb, detector, classifier, classify_tf, device)

                human_total += sum(1 for _, l, _ in results if l == "HUMAN")
                animal_total += sum(1 for _, l, _ in results if l == "ANIMAL")

                frame = draw_detections(frame, results)
                out_writer.write(frame)

                if frame_idx % 5 == 0 or frame_idx == max_frames:
                    progress.progress(
                        frame_idx / max_frames,
                        text=f"Frame {frame_idx}/{max_frames}"
                    )

            cap.release()
            out_writer.release()
            progress.empty()

            st.success(f"Processed {frame_idx} frames")

            c1, c2 = st.columns(2)
            c1.metric("Human detections (total)", human_total)
            c2.metric("Animal detections (total)", animal_total)

            with st.spinner("Encoding video for playback..."):
                h264_path = reencode_for_browser(tmp_output.name)

            with open(h264_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)

            out_name = os.path.basename(video_path).rsplit(".", 1)[0]
            st.download_button(
                "Download annotated video",
                data=video_bytes,
                file_name=f"annotated_{out_name}.mp4",
                mime="video/mp4",
            )

            # Cleanup temp files
            for p in [tmp_output.name, h264_path]:
                if os.path.exists(p):
                    os.unlink(p)
            # Only delete tmp_input if it was an upload (not a sample)
            if input_mode == "Upload a video" and os.path.exists(video_path):
                os.unlink(video_path)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: OCR
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ocr:
    st.header("Offline OCR — Stenciled / Industrial Text")
    st.caption("Pick a sample image or upload your own to extract text.")

    sample_images = list_sample_images()

    ocr_mode = st.radio(
        "Input source", ["Select from samples", "Upload an image"],
        horizontal=True, key="ocr_mode"
    )

    img_original = None
    source_name = None

    if ocr_mode == "Select from samples" and sample_images:
        chosen_img = st.selectbox("Sample images", sample_images, key="ocr_sample")
        img_path = os.path.join("test_images", chosen_img)
        img_original = cv2.imread(img_path)
        source_name = chosen_img

        # Show preview
        if img_original is not None:
            st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB), width=400, caption=chosen_img)

    elif ocr_mode == "Upload an image":
        uploaded_img = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg", "bmp", "webp", "tiff"], key="ocr_upload"
        )
        if uploaded_img is not None:
            file_bytes = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
            img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            source_name = uploaded_img.name
    elif ocr_mode == "Select from samples" and not sample_images:
        st.info("No sample images found in `test_images/`. Upload one instead.")

    if img_original is not None and source_name is not None:
        run_btn = st.button("Run OCR", key="run_ocr", type="primary")

        if run_btn:
            ocr = load_ocr()
            ori_engine = load_orientation_detector()

            with st.spinner("Running OCR pipeline..."):
                img, scale = resize_if_needed(img_original)
                detections, angle = run_with_orientation_check(ocr, ori_engine, img)

                if angle != 0:
                    img_original = rotate_image(img_original, angle)
                    st.info(f"Detected {angle}° rotation — auto-corrected")

                if scale < 1.0:
                    for d in detections:
                        d["bbox"] = [[int(p[0] / scale), int(p[1] / scale)] for p in d["bbox"]]

                annotated = ocr_draw_results(img_original, detections)

            col_img, col_text = st.columns([3, 2])

            with col_img:
                st.subheader("Detected Text Regions")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

            with col_text:
                st.subheader("Extracted Text")

                if not detections:
                    st.warning("No text detected.")
                else:
                    for d in detections:
                        st.markdown(
                            f"**`{d['text']}`** &nbsp;— confidence: `{d['confidence']:.2f}`"
                        )

                    st.divider()
                    full_text = " | ".join(d["text"] for d in detections)
                    st.text_area("Full text (combined)", full_text, height=80)

                    result_json = {
                        "file": source_name,
                        "num_text_regions": len(detections),
                        "extracted_lines": [
                            {"text": d["text"], "confidence": d["confidence"], "bbox": d["bbox"]}
                            for d in detections
                        ],
                        "full_text": full_text,
                    }
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(result_json, indent=2),
                        file_name=f"{source_name.rsplit('.', 1)[0]}_ocr.json",
                        mime="application/json",
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: RESULTS BROWSER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_results:
    st.header("Results Browser")
    st.caption("Browse previously generated detection videos and OCR results from `outputs/`.")

    # ── Detection Results ─────────────────────────────────────────────────────

    st.subheader("Detection Videos")

    det_videos = list_detection_videos()

    if not det_videos:
        st.info("No detection videos found in `outputs/`. Run detection first to generate results.")
    else:
        chosen_vid = st.selectbox(
            "Select a detection video",
            det_videos,
            format_func=os.path.basename,
            key="results_det_video",
        )

        if chosen_vid:
            cap = cv2.VideoCapture(chosen_vid)
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Skip corrupt files (missing moov atom, truncated writes, etc.)
            if vid_fps == 0 or vid_frames == 0:
                st.warning(
                    f"`{os.path.basename(chosen_vid)}` appears corrupt and cannot be played. "
                    "This usually happens when a previous detection run was interrupted. "
                    "Re-run detection to regenerate it."
                )
            else:
                vid_duration = round(vid_frames / vid_fps, 1)

                st.markdown(
                    f"**{os.path.basename(chosen_vid)}** &nbsp;|&nbsp; "
                    f"{vid_w}x{vid_h} &nbsp;|&nbsp; {vid_fps:.0f} FPS &nbsp;|&nbsp; "
                    f"{vid_duration}s &nbsp;|&nbsp; {format_file_size(chosen_vid)}"
                )

                try:
                    h264_cached = get_cached_h264_video(chosen_vid)
                    with open(h264_cached, "rb") as f:
                        vid_bytes = f.read()
                    st.video(vid_bytes)

                    st.download_button(
                        "Download video",
                        data=vid_bytes,
                        file_name=os.path.basename(chosen_vid),
                        mime="video/mp4",
                        key="results_det_download",
                    )
                except subprocess.CalledProcessError:
                    st.error("Failed to re-encode video for browser playback. Is ffmpeg installed?")

    st.divider()

    # ── OCR Results ───────────────────────────────────────────────────────────

    st.subheader("OCR Results")

    ocr_pairs = list_ocr_results()

    if not ocr_pairs:
        st.info("No OCR results found in `outputs/`. Run OCR first to generate results.")
    else:
        # Build display names from the jpg filenames
        display_names = [os.path.basename(jpg) for jpg, _ in ocr_pairs]
        chosen_idx = st.selectbox(
            "Select an OCR result",
            range(len(ocr_pairs)),
            format_func=lambda i: display_names[i],
            key="results_ocr_select",
        )

        ocr_jpg, ocr_json = ocr_pairs[chosen_idx]

        with open(ocr_json, "r") as f:
            ocr_data = json.load(f)

        col_img, col_text = st.columns([3, 2])

        with col_img:
            st.markdown("**Annotated Image**")
            annotated_img = cv2.imread(ocr_jpg)
            if annotated_img is not None:
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), width="stretch")
            else:
                st.error(f"Could not load image: {ocr_jpg}")

        with col_text:
            st.markdown("**Extracted Text**")

            lines = ocr_data.get("extracted_lines", [])
            if not lines:
                st.warning("No text regions in this result.")
            else:
                for line in lines:
                    st.markdown(
                        f"**`{line['text']}`** &nbsp;— confidence: `{line['confidence']:.2f}`"
                    )

                st.divider()
                full_text = ocr_data.get("full_text", "")
                st.text_area("Full text (combined)", full_text, height=80, key="results_ocr_text")

            with st.expander("Raw JSON"):
                st.json(ocr_data)

            json_bytes = json.dumps(ocr_data, indent=2)
            st.download_button(
                "Download JSON",
                data=json_bytes,
                file_name=os.path.basename(ocr_json),
                mime="application/json",
                key="results_ocr_download",
            )
