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

## Run with Docker
```bash
docker build -t onnx-fastapi:cpu .
docker run --rm -p 8000:8000 onnx-fastapi:cpu
# open http://localhost:8000/docs
