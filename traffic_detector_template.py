"""
Traffic Detector & Instance Segmentation Template
File: traffic_detector_template.py

Requirements:
  - Python 3.8+
  - pip install ultralytics opencv-python numpy
    (ultralytics provides YOLOv8 including instance segmentation: yolov8x-seg.pt)

What this script does:
  - Loads a YOLOv8 segmentation model (pretrained or custom)
  - Processes an image, video, or webcam stream
  - Draws instance masks, bounding boxes and class labels
  - Keeps a running class-wise count per frame and overlays counts
  - Writes output to disk (video) if requested

How to run (examples):
  python traffic_detector_template.py --source traffic.mp4 --output out.mp4
  python traffic_detector_template.py --source 0            # webcam
  python traffic_detector_template.py --source img.jpg --show

If you want to train on custom data, label with LabelMe/Roboflow in YOLO format and run:
  yolo detect train data=data.yaml model=yolov8m-seg.pt epochs=100

Notes:
  - This is a minimal, single-file template meant for quick testing and extension.
  - For production: add batching, queue-based IO, thread-safe writer, FPS smoothing, and model warmstart.

"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', '-s', type=str, required=True,
                   help='Path to image/video file or integer webcam id (0)')
    # p.add_argument('--model', '-m', type=str, default='yolov8x-seg.pt',
    #                help='Path to YOLOv8 segmentation model')
    p.add_argument('--model', '-m', type=str, default='yolov8s-seg.pt',
                   help='Path to YOLOv8 segmentation model (try n/s/m/l/x-seg.pt for speed/accuracy tradeoff)')
    p.add_argument('--output', '-o', type=str, default=None,
                   help='Path to output video (optional)')
    p.add_argument('--show', action='store_true', help='Show window')
    p.add_argument('--device', type=str, default='cpu', help='cpu or gpu device id like 0')
    return p.parse_args()


# Simple color palette generator
def random_colors(n, seed=42):
    np.random.seed(seed)
    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(n)]
    return colors


def draw_mask(image, mask, color, alpha=0.35):
    """Overlay a segmentation mask onto image. mask is boolean 2D array."""
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    colored = np.zeros_like(image, dtype=np.uint8)
    colored[mask] = color
    # alpha blend
    cv2.addWeighted(colored, alpha, image, 1 - alpha, 0, dst=image)


def visualize_result(frame, result, names, colors):
    """Draw boxes, labels and masks on a frame using a single ultralytics Result object."""
    h, w = frame.shape[:2]

    # draw masks first so boxes/labels are on top
    if hasattr(result, 'masks') and result.masks is not None:
        # result.masks.data is (N, H, W) with floats 0/1 in some versions; other versions supply r.masks.xy
        try:
            masks = result.masks.data.cpu().numpy()  # (N, H, W)
        except Exception:
            # fallback: some versions provide r.masks.numpy()
            masks = result.masks.cpu().numpy()

        for i, mask in enumerate(masks):
            cls = int(result.boxes.cls[i].cpu().numpy()) if len(result.boxes) > 0 else 0
            color = colors[cls % len(colors)]
            # mask may be smaller than frame; resize if needed
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask.astype('uint8') * 255, (w, h))
                mask = mask > 127
            else:
                mask = mask > 0.5
            draw_mask(frame, mask, color)

    # draw boxes and labels
    if hasattr(result, 'boxes') and result.boxes is not None:
        for i, box in enumerate(result.boxes):
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # [x1,y1,x2,y2]
            x1, y1, x2, y2 = xyxy.tolist()
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0]) if hasattr(box, 'conf') else 0.0
            label = f"{names[cls]} {conf:.2f}"
            color = colors[cls % len(colors)]
            # box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def count_classes(result, names):
    """Return a dict of counts per class in this result."""
    counts = {}
    if result.boxes is None or len(result.boxes) == 0:
        return counts
    try:
        classes = result.boxes.cls.cpu().numpy().astype(int).flatten()
    except Exception:
        classes = np.array([int(x) for x in result.boxes.cls])
    for c in classes:
        name = names[c]
        counts[name] = counts.get(name, 0) + 1
    return counts


def overlay_counts(frame, counts):
    """Put class counts overlay on top-left corner."""
    x0, y0 = 10, 10
    i = 0
    for k, v in counts.items():
        text = f"{k}: {v}"
        y = y0 + i * 20 + 15
        cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        i += 1
    return frame


def main():
    args = parse_args()

    # interpret source
    source = args.source
    try:
        src_id = int(source)
        cap = cv2.VideoCapture(src_id)
    except Exception:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print('Error: Cannot open source', source)
        return

    model = YOLO(args.model)
    names = model.names if hasattr(model, 'names') else {0: 'object'}

    # prepare output writer if requested
    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # prepare colors for classes
    max_cls = max([int(k) for k in names.keys()]) + 1
    colors = random_colors(max_cls)

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # run inference; stream=True yields generator of results per image
        # using model.predict or model(frame, stream=True) works. We'll use model.predict for clarity.
        results = model(frame)
        # results is list-like; for single image we take first
        r = results[0]

        # count classes
        counts = count_classes(r, names)

        # Create frames for both views
        original = frame.copy()
        detection_frame = frame.copy()

        # Draw boxes and labels on original (no masks)
        if hasattr(r, 'boxes') and r.boxes is not None:
            for i, box in enumerate(r.boxes):
                xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                x1, y1, x2, y2 = xyxy.tolist()
                cls = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0]) if hasattr(box, 'conf') else 0.0
                label = f"{names[cls]} {conf:.2f}"
                color = colors[cls % len(colors)]
                # box
                cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
                # label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(original, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                cv2.putText(original, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add counts to original video
        overlay_counts(original, counts)
        
        # Create full detection view with masks
        visualize_result(detection_frame, r, names, colors)
        overlay_counts(detection_frame, counts)

        # show fps and frame idx on both frames
        elapsed = time.time() - start_time
        fps_text = f"Frame: {frame_idx}  FPS: {frame_idx/elapsed:.2f}"
        cv2.putText(original, fps_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(detection_frame, fps_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if args.show:
            # Show both windows side by side
            cv2.imshow('Original Video', original)
            cv2.imshow('Traffic Detector (Live)', detection_frame)
            # Check for 'q' press on either window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer is not None:
            writer.write(detection_frame)  # Save the detection view

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()
