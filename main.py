import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms, models
from torchvision.ops import nms
from PIL import Image
import cv2
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.6  # Higher = fewer false positives
NMS_THRESHOLD = 0.3  # Lower = fewer duplicate boxes 


PROCESS_SECONDS = 3  

print(f"Using device: {DEVICE}")
if PROCESS_SECONDS:
    print(f"⚡ QUICK TEST MODE: Processing only first {PROCESS_SECONDS} seconds")

print("Loading models...")

# MODEL 1: Faster R-CNN (Detection)
detector = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 3
in_features = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
detector.load_state_dict(torch.load('models/detector_final.pth', map_location=DEVICE))
detector = detector.to(DEVICE)
detector.eval()
print("✅ Detector loaded")

# MODEL 2: EfficientNet (Classifier)
classifier = models.efficientnet_b3()
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 2)
classifier.load_state_dict(torch.load('models/classifier_best.pth', map_location=DEVICE))
classifier = classifier.to(DEVICE)
classifier.eval()
print("✅ Classifier loaded")

# Transform for classifier
classify_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def apply_nms(boxes, scores, labels, nms_threshold=NMS_THRESHOLD):
    """
    Apply Non-Maximum Suppression to remove duplicate detections
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    # Apply NMS per class
    keep_indices = []
    
    for class_id in labels.unique():
        # Get boxes for this class
        class_mask = labels == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        # Apply NMS
        keep = nms(class_boxes, class_scores, nms_threshold)
        
        # Convert back to original indices
        original_indices = torch.where(class_mask)[0]
        keep_indices.extend(original_indices[keep].tolist())
    
    keep_indices = torch.tensor(keep_indices)
    
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]

def process_video(video_path, output_path):
    """
    Two-stage pipeline with NMS:
    1. Detect objects with Faster R-CNN + NMS
    2. Classify each detection with EfficientNet
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open {video_path}")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if PROCESS_SECONDS:
        max_frames = int(fps * PROCESS_SECONDS)
        total_frames = min(total_frames, max_frames)
        print(f"Processing first {PROCESS_SECONDS} seconds ({total_frames} frames)...")
    else:
        print(f"Processing entire video ({total_frames} frames)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if PROCESS_SECONDS and frame_count > total_frames:
            print(f"\n⏹️  Stopped at {PROCESS_SECONDS} seconds")
            break
        
        if frame_count % 10 == 0:
            print(f"  Frame {frame_count}/{total_frames}", end='\r')
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        img_tensor = transforms.ToTensor()(rgb_frame).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            detections = detector(img_tensor)[0]
        
        boxes = detections['boxes']
        scores = detections['scores']
        labels = detections['labels']
    
        keep = scores > CONFIDENCE_THRESHOLD
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        if len(boxes) > 0:
            boxes, scores, labels = apply_nms(boxes, scores, labels, NMS_THRESHOLD)

        for box, det_score, det_label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box.tolist())
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
        
            crop = rgb_frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            

            crop_pil = Image.fromarray(crop)
            crop_tensor = classify_transform(crop_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred = classifier(crop_tensor)
                class_prob = torch.softmax(pred, dim=1)
                class_idx = pred.argmax(1).item()
                clf_conf = class_prob[0][class_idx].item()
            

            label_name = "ANIMAL" if class_idx == 0 else "HUMAN"
            

            if label_name == "ANIMAL":
                color = (255, 140, 0)  
            else:
                color = (0, 255, 0)    
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            combined_conf = det_score.item() * clf_conf
            text = f"{label_name} {combined_conf:.2f}"
            
            # Text background
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 15),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
        
            cv2.putText(
                frame,
                text,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"\n✅ Processed {frame_count} frames")

def process_all_videos():
    os.makedirs("test_videos", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    video_files = [f for f in os.listdir("test_videos") 
                   if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print("⚠️  No videos in ./test_videos/")
        return
    
    print(f"\nFound {len(video_files)} video(s)\n")
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Video {i}/{len(video_files)}: {video_file}")
        print('='*60)
        
        input_path = os.path.join("test_videos", video_file)
        output_path = os.path.join("outputs", f"annotated_{video_file}")
        
        process_video(input_path, output_path)
        print(f"✅ Saved: {output_path}")
    
    print(f"\n{'='*60}")
    print("ALL VIDEOS PROCESSED!")
    print('='*60)

if __name__ == "__main__":
    process_all_videos()