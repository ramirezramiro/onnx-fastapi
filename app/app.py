from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort
from PIL import Image
import numpy as np
import io, os, time, pathlib, urllib.request

MODEL_PATH = os.getenv("MODEL_PATH", "app/model.onnx")

def ensure_model(path=MODEL_PATH):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        url = "https://raw.githubusercontent.com/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
        urllib.request.urlretrieve(url, str(p))
        print(f"Downloaded fallback model to {p}")
    return str(p)
MODEL_PATH = ensure_model()

IMG_SIZE = 224
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2,0,1))  # HWC->CHW
    return np.expand_dims(arr, 0)     # NCHW

app = FastAPI(title="ONNX Runtime Inference API")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Invalid image")
    x = preprocess(img)
    t0 = time.time()
    outputs = sess.run(None, {"input": x})
    dt_ms = (time.time() - t0) * 1000
    logits = outputs[0][0]
    top5_idx = np.argsort(logits)[-5:][::-1].tolist()
    top5_score = [float(logits[i]) for i in top5_idx]
    return JSONResponse({"top5_idx": top5_idx, "top5_score": top5_score, "avg_ms": round(dt_ms, 2)})
