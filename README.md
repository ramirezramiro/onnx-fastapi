**Prerequisites**
WSL 2 (Windows Subsystem for Linux) → install from Microsoft Store:
https://apps.microsoft.com/detail/9p9tqf7mrm4r?hl=en-US&gl=TW

**CPU-optimized inference (ONNX Runtime)**
Model: MobileNetV3-Small (ImageNet)
Hardware: Colab CPU
Latency (avg): ~4.23 ms/img
Throughput: ~236.27 FPS
Export: PyTorch → ONNX (opset 13, dynamic batch), ORT graph optimizations ON

# onnx-fastapi (CPU)

Minimal FastAPI service for ONNX inference on CPU (MobileNetV3-Small example).

## Run and Build (Docker)
```bash
docker build -t onnx-fastapi:cpu .
docker run --rm -p 8000:8000 onnx-fastapi:cpu
# open http://localhost:8000/docs
```

## API Health check
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## Prediction
```bash
# Replace with your image path
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@samples/cat.jpg"
```
## Prediction
```json
{
  "top1": {"label": "tabby_cat", "prob": 0.87},
  "top5": [
    {"label": "tabby_cat", "prob": 0.87},
    {"label": "tiger_cat", "prob": 0.07},
    {"label": "Egyptian_cat", "prob": 0.03},
    {"label": "lynx", "prob": 0.02},
    {"label": "cougar", "prob": 0.01}
  ]
}
```

## Architecture Diagram
```mermaid
flowchart LR
  A[Client / cURL / App] -->|HTTP: /predict| B[FastAPI Service]
  B -->|NumPy tensors| C[ONNX Runtime]
  C -->|CPU| D[(Model: model.onnx)]

  subgraph Docker Container
    B
    C
    D
  end

  E[(Host FS)]
  D ---|bind-mount (optional)| E
```


## Final takeaway 

FastAPI exposes /health and /predict.

ONNX Runtime (CPU) runs inference—no GPU required.

Model can be baked into the image or bind-mounted at runtime for quick swaps.