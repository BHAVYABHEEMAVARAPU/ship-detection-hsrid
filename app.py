import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="ShipVision SAR", page_icon="🚢", layout="wide")
st.title("🚢 ShipVision SAR")
st.caption("SAR ship detection using your trained YOLOv8-Nano model")

MODEL_PATH = Path(__file__).parent / "best.pt"

@st.cache_resource
def load_model(path):
    return YOLO(path)

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
    imgsz = st.select_slider("Inference image size",
                             options=[640, 704, 768, 800, 896, 1024],
                             value=800)

if not MODEL_PATH.exists():
    st.error("Place your trained checkpoint named best.pt in the same folder as app.py.")
    st.code("shipvision_streamlit/\n├── app.py\n├── requirements.txt\n└── best.pt")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load best.pt: {exc}")
    st.stop()

uploaded = st.file_uploader("Upload a SAR image",
                            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if uploaded is None:
    st.info("Upload a SAR image to detect ships.")
else:
    try:
        pil_image = Image.open(uploaded).convert("RGB")
        image = np.array(pil_image)

        with st.spinner("Detecting ships..."):
            result = model.predict(source=image, conf=confidence, iou=iou,
                                   imgsz=imgsz, verbose=False)[0]

        annotated = result.plot()[:, :, ::-1]
        boxes = result.boxes
        rows = []

        if boxes is not None and len(boxes):
            coords = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for i, (box, score, class_id) in enumerate(zip(coords, scores, classes), 1):
                rows.append({
                    "Detection": i,
                    "Class": model.names.get(class_id, str(class_id)),
                    "Confidence": round(float(score), 4),
                    "X1": round(float(box[0]), 1),
                    "Y1": round(float(box[1]), 1),
                    "X2": round(float(box[2]), 1),
                    "Y2": round(float(box[3]), 1),
                })

        left, right = st.columns(2)
        with left:
            st.subheader("Input image")
            st.image(image, use_container_width=True)
        with right:
            st.subheader("Detected ships")
            st.image(annotated, use_container_width=True)

        st.divider()
        a, b, c = st.columns(3)
        a.metric("Ships detected", len(rows))
        b.metric("Confidence threshold", f"{confidence:.2f}")
        c.metric("Inference image size", f"{imgsz}px")

        if rows:
            df = pd.DataFrame(rows)
            st.subheader("Detection details")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download detections CSV",
                               df.to_csv(index=False).encode("utf-8"),
                               "ship_detections.csv", "text/csv")
        else:
            st.warning("No ships detected. Try lowering the confidence threshold.")

        buffer = io.BytesIO()
        Image.fromarray(annotated).save(buffer, format="PNG")
        st.download_button("Download annotated image", buffer.getvalue(),
                           "ship_detection_result.png", "image/png")

        st.caption(
            "Research prototype: predictions are not verified labels. "
            "Performance may vary on SAR imagery from sensors or conditions "
            "not represented in HRSID."
        )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
