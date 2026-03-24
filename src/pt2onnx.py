import os
import torch

def convert_pt_to_onnx(pt_file_path, input_size, onnx_file_path):
    if not os.path.exists(pt_file_path):
        print(f"Error: PyTorch model file does not exist: {pt_file_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_obj = torch.load(pt_file_path, map_location=device)

    # Some .pt files are checkpoints (dict), not nn.Module.
    if isinstance(loaded_obj, torch.nn.Module):
        model = loaded_obj
    elif isinstance(loaded_obj, dict):
        if "model" in loaded_obj and isinstance(loaded_obj["model"], torch.nn.Module):
            model = loaded_obj["model"]
        elif "ema" in loaded_obj and isinstance(loaded_obj["ema"], torch.nn.Module):
            model = loaded_obj["ema"]
        else:
            raise TypeError(
                "Loaded checkpoint is a dict without a usable nn.Module in key 'model' or 'ema'."
            )
    else:
        raise TypeError(f"Unsupported object type from torch.load: {type(loaded_obj)}")

    model = model.to(device)

    # Match dummy input dtype with model weights to avoid dtype mismatch during tracing.
    model_dtype = next(model.parameters()).dtype if any(True for _ in model.parameters()) else torch.float32

    # Export as fp32 for better ONNX/ONNX Runtime compatibility.
    if model_dtype == torch.float16:
        print("Model weights are float16; casting to float32 for ONNX export compatibility.")
        model = model.float()
        model_dtype = torch.float32

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        input_size[0],
        input_size[1],
        device=device,
        dtype=model_dtype,
    )
    print(f"Dummy input shape: {tuple(dummy_input.shape)}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )

    print(f"ONNX model has been saved to: {onnx_file_path}")
    
if __name__ == "__main__":
    pt_file_path = "/home/owen/桌面/torch learning/yolo11n.pt"
    input_size = (640, 640)
    onnx_file_path = "/home/owen/桌面/torch learning/yolo11n.onnx"
    convert_pt_to_onnx(pt_file_path, input_size, onnx_file_path)