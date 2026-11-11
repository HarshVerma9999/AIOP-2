#!/usr/bin/env python3
# anpr.py: detect and read vehicle number plates in an image (YOLO detector + fast-plate-ocr reader)
import argparse, json, re, sys
import cv2, numpy as np
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

def crop_with_margin(img, x0, y0, x1, y1, pad=0.05):
    h, w = img.shape[:2]; dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return img[int(max(0, y0 - dy)):int(min(h, y1 + dy)), int(max(0, x0 - dx)):int(min(w, x1 + dx))]

def main():
    ap = argparse.ArgumentParser(description="Detect and read number plates in an image")
    ap.add_argument("--weights", required=True); ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--review-threshold", type=float, default=0.80); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    img = cv2.imread(a.source)
    if img is None: sys.exit(f"Cannot read image: {a.source}")
    det = YOLO(a.weights)
    ocr = LicensePlateRecognizer("cct-s-v2-global-model", device="cpu")
    rgb = "rgb" in str(ocr.config.image_color_mode).lower()
    res = det.predict(img, conf=a.conf, device=a.device, verbose=False)[0]
    results = []
    for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        crop = crop_with_margin(img, *b)
        if min(crop.shape[:2]) < 4: continue
        p = ocr.run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB if rgb else cv2.COLOR_BGR2GRAY), return_confidence=True)[0]
        text = re.sub(r"[^A-Z0-9]", "", p.plate.upper())
        conf = float(np.min(p.char_probs)) if p.char_probs is not None and np.size(p.char_probs) else 0.0
        results.append({"plate": text, "det_conf": round(float(c), 3), "ocr_conf": round(conf, 3),
                        "bbox": [int(v) for v in b], "needs_review": conf < a.review_threshold or len(text) < 4})
        x0, y0, x1, y1 = map(int, b); col = (0, 165, 255) if results[-1]["needs_review"] else (0, 200, 0)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        cv2.putText(img, text, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    if a.out: cv2.imwrite(a.out, img)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
