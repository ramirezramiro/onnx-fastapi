import pathlib
import torch, torchvision, onnx

def main():
    model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    out = pathlib.Path("app/model.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    onnx.checker.check_model(onnx.load(str(out)))
    print(f"Exported OK -> {out.resolve()}")

if __name__ == "__main__":
    main()
