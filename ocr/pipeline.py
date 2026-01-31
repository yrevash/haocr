import cv2
import numpy as np
import json
import os
from paddleocr import PaddleOCR

MAX_SIDE = 1280  # Cap images here before any processing — prevents PaddleOCR
                  # from allocating massive feature maps on high-res photos

def resize_if_needed(img):
    """
    Downscale to MAX_SIDE on the longest edge. OCR accuracy doesn't improve
    much past ~1280px for stenciled text, but memory usage grows quadratically.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= MAX_SIDE:
        return img, 1.0
    scale = MAX_SIDE / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def enhance_image(img):
    """
    Returns multiple enhanced versions of the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE pulls out text that's nearly invisible on dark surfaces .
    # clipLimit=3.0 is a sweet spot for military boxes where paint has faded
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Bilateral filter removes surface texture/rust noise while keeping
    # the hard edges of stenciled letters intact. Regular gaussian blur
    # would smear the letter edges.
    denoised = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
    # Sharpen to recover faded stencil edges — stencil paint often bleeds
    blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    # PaddleOCR's detection model uses color features, so a contrast-boosted color image often catches things the grayscale versions miss.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    color_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return {
        "original": img,
        "clahe_gray": cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR),
        "sharpened": cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR),
        "color_enhanced": color_enhanced,
    }


def build_ocr_engine():
    ocr = PaddleOCR(
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        lang="en",
        text_det_thresh=0.2,
        text_det_box_thresh=0.3,
    )
    return ocr


def build_orientation_detector():
    """
    Separate engine just for detecting image rotation angle.
    We don't use this engine for actual OCR — its text results are fragmented
    because the internal rotation messes with the detection grid. We only
    read the angle it detects, then rotate the image ourselves.
    """
    return PaddleOCR(
        use_textline_orientation=True,
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        lang="en",
    )


def detect_orientation(ori_engine, img):
    """Returns the detected rotation angle (0, 90, 180, or 270)."""
    for res in ori_engine.predict(img):
        doc_res = res.get("doc_preprocessor_res", {})
        return doc_res.get("angle", 0)
    return 0


def rotate_image(img, angle):
    """Rotate image by 0/90/180/270 degrees."""
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def avg_confidence(detections):
    if not detections:
        return 0.0
    return sum(d["confidence"] for d in detections) / len(detections)


def run_ocr_on_variants(ocr, variants):
    all_detections = []

    for name, img in variants.items():
        for res in ocr.predict(img):
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("dt_polys", [])

            for text, score, poly in zip(texts, scores, polys):
                # Skiping garbage low-confidence hits are almost always
                # surface scratches or rust patterns misread as text
                if score < 0.3 or len(text.strip()) == 0:
                    continue

                bbox = poly.tolist() if hasattr(poly, "tolist") else poly
                all_detections.append({
                    "text": text.strip(),
                    "confidence": round(float(score), 4),
                    "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                    "source_variant": name,
                })

    # If we find two variants found the same text in roughly the same spot  keep the higher-confidence one
    merged = deduplicate(all_detections)
    # Single/double-char detections need high confidence to survive
    merged = [
        d for d in merged
        if (len(d["text"]) > 2) or (len(d["text"]) > 1 and d["confidence"] > 0.6) or (d["confidence"] > 0.9)
    ]
    merged = remove_absorbed_fragments(merged)

    return merged


def deduplicate(detections, iou_thresh=0.3):
    """
    Remove duplicate detections across enhancement variants.
    Keeps the highest-confidence version of each text region.
    """
    if not detections:
        return []

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    keep = []

    for det in detections:
        is_dup = False
        for kept in keep:
            if box_iou(det["bbox"], kept["bbox"]) > iou_thresh:
                if text_similarity(det["text"], kept["text"]):
                    is_dup = True
                    break
        if not is_dup:
            keep.append(det)

    return keep


def box_iou(box1, box2):
    """IoU between two quadrilateral boxes (approximated as axis-aligned rects)."""
    def to_rect(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return min(xs), min(ys), max(xs), max(ys)

    r1 = to_rect(box1)
    r2 = to_rect(box2)

    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[2], r2[2])
    y2 = min(r1[3], r2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
    area2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def text_similarity(t1, t2):
    """
    Check if two strings are roughly the same. OCR on stenciled text often
    produces slightly different reads of the same region (e.g., "TEAM" vs "TEAN")
    so we need fuzzy matching, not exact.
    """
    t1, t2 = t1.lower().strip(), t2.lower().strip()
    if t1 == t2:
        return True
    if t1 in t2 or t2 in t1:
        return True
    t1_nospace = t1.replace(" ", "")
    t2_nospace = t2.replace(" ", "")
    if t1_nospace == t2_nospace:
        return True
    if t1_nospace in t2_nospace or t2_nospace in t1_nospace:
        return True
    if len(t1) > 1 and len(t2) > 1:
        shorter, longer = sorted([t1, t2], key=len)
        matches = sum(1 for c1, c2 in zip(shorter, longer) if c1 == c2)
        ratio = matches / max(len(shorter), 1)
        if ratio > 0.6:
            return True
    return False


def remove_absorbed_fragments(detections):
    if len(detections) < 2:
        return detections

    keep = []
    for i, det in enumerate(detections):
        absorbed = False
        for j, other in enumerate(detections):
            if i == j:
                continue
            if len(other["text"]) <= len(det["text"]):
                continue
            containment = box_contained(det["bbox"], other["bbox"])
            if det["text"].lower() in other["text"].lower() and containment > 0.3:
                absorbed = True
                break
            # Box is almost entirely inside a larger box  it's a word-level fragment of a line-level detection, even if OCR read it differently
            if containment > 0.7:
                absorbed = True
                break
        if not absorbed:
            keep.append(det)

    return keep


def box_contained(inner_box, outer_box):
    """What fraction of the inner box's area is inside the outer box."""
    def to_rect(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return min(xs), min(ys), max(xs), max(ys)

    r1 = to_rect(inner_box)
    r2 = to_rect(outer_box)

    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[2], r2[2])
    y2 = min(r1[3], r2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (r1[2] - r1[0]) * (r1[3] - r1[1])

    return inter / inner_area if inner_area > 0 else 0

def draw_results(img, detections):
    """Draw detected text boxes and labels on the image."""
    out = img.copy()
    for det in detections:
        pts = np.array(det["bbox"], dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)

        x, y = pts[0]
        label = f"{det['text']} ({det['confidence']:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw, y), (0, 255, 0), -1)
        cv2.putText(out, label, (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out

def build_structured_output(image_path, detections):
    return {
        "file": os.path.basename(image_path),
        "num_text_regions": len(detections),
        "extracted_lines": [
            {
                "text": d["text"],
                "confidence": d["confidence"],
                "bbox": d["bbox"],
            }
            for d in detections
        ],
        "full_text": " | ".join(d["text"] for d in detections),
    }


def run_with_orientation_check(ocr, ori_engine, img):
    """
    Auto-detect + validate: run OCR normally, then check if the image might
    be rotated. If the orientation detector says it's rotated, manually rotate
    and re-run, then keep whichever result set has higher avg confidence.
    """
    # Pass 1: run on image as-is
    variants_normal = enhance_image(img)
    dets_normal = run_ocr_on_variants(ocr, variants_normal)

    # Quick orientation check — only uses one small forward pass
    angle = detect_orientation(ori_engine, img)

    if angle == 0:
        return dets_normal, 0  # no rotation needed

    # Pass 2: rotate and re-run
    rotated = rotate_image(img, angle)
    variants_rotated = enhance_image(rotated)
    dets_rotated = run_ocr_on_variants(ocr, variants_rotated)

    # Keep whichever is better — this is the "validate" step
    # that protects against the orientation detector being wrong
    conf_normal = avg_confidence(dets_normal)
    conf_rotated = avg_confidence(dets_rotated)

    if conf_rotated > conf_normal and len(dets_rotated) >= len(dets_normal):
        return dets_rotated, angle
    else:
        return dets_normal, 0


def process_image(image_path, ocr, output_dir="outputs", ori_engine=None):
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"[ERROR] Could not read: {image_path}")
        return None

    print(f"\n[OCR] Processing: {os.path.basename(image_path)}")

    # Resize before anything else to keep memory under control
    img, scale = resize_if_needed(img_original)
    if scale < 1.0:
        h, w = img.shape[:2]
        print(f"  Resized: {img_original.shape[1]}x{img_original.shape[0]} -> {w}x{h}")

    if ori_engine is not None:
        detections, angle = run_with_orientation_check(ocr, ori_engine, img)
        if angle != 0:
            print(f"  Detected rotation: {angle} deg — corrected")
            img_original = rotate_image(img_original, angle)
    else:
        variants = enhance_image(img)
        detections = run_ocr_on_variants(ocr, variants)

    # Scale bounding boxes back to original image coords so annotations
    # line up when drawn on the full-res image
    if scale < 1.0:
        for d in detections:
            d["bbox"] = [[int(p[0] / scale), int(p[1] / scale)] for p in d["bbox"]]

    result = build_structured_output(image_path, detections)

    print(f"  Found {len(detections)} text region(s):")
    for d in detections:
        print(f"    \"{d['text']}\"  (conf: {d['confidence']:.2f})")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    annotated = draw_results(img_original, detections)
    ann_path = os.path.join(output_dir, f"{base}_ocr.jpg")
    cv2.imwrite(ann_path, annotated)

    json_path = os.path.join(output_dir, f"{base}_ocr.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved: {ann_path}")
    print(f"  Saved: {json_path}")

    return result


def process_folder(input_dir, output_dir="outputs"):
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ])

    if not files:
        print(f"[ERROR] No images found in {input_dir}")
        return []

    print(f"[OCR] Found {len(files)} image(s) in '{input_dir}'")

    ocr = build_ocr_engine()
    ori_engine = build_orientation_detector()
    all_results = []

    for f in files:
        path = os.path.join(input_dir, f)
        result = process_image(path, ocr, output_dir, ori_engine=ori_engine)
        if result:
            all_results.append(result)

    combined_path = os.path.join(output_dir, "ocr_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OCR] Combined results saved to: {combined_path}")

    return all_results




if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"

    results = process_folder(input_dir, output_dir)

    print("\n" + "=" * 50)
    print("  OCR SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"\n  {r['file']}:")
        print(f"    -> {r['full_text']}")
