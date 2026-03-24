import argparse

import torch


def parse_args():
	parser = argparse.ArgumentParser(description="Inspect YOLO11 model structure")
	parser.add_argument(
		"--weights",
		type=str,
		default="yolo11n.pt",
		help="YOLO11 weights file or model name, e.g. yolo11n.pt",
	)
	parser.add_argument(
		"--imgsz",
		type=int,
		default=640,
		help="Random input image size (square)",
	)
	return parser.parse_args()


def print_header(torch_version, ultralytics_version, weights_name):
	print("=" * 80)
	print("Environment")
	print("-" * 80)
	print(f"torch: {torch_version}")
	print(f"ultralytics: {ultralytics_version}")
	print(f"weights: {weights_name}")


def print_model_info(yolo_model):
	core_model = yolo_model.model
	print("\n" + "=" * 80)
	print("Model Overview")
	print("-" * 80)
	print(core_model)

	total_params = sum(p.numel() for p in core_model.parameters())
	trainable_params = sum(p.numel() for p in core_model.parameters() if p.requires_grad)
	print("\nParameter Stats")
	print(f"total params: {total_params:,}")
	print(f"trainable params: {trainable_params:,}")

	print("\nKey Modules")
	print(f"model type: {type(core_model).__name__}")
	if hasattr(core_model, "model"):
		print(f"inner blocks: {type(core_model.model).__name__}")


def summarize_output(output, prefix="output"):
	if torch.is_tensor(output):
		print(f"{prefix} tensor shape: {tuple(output.shape)}")
		return

	if isinstance(output, dict):
		print(f"{prefix} dict keys: {list(output.keys())}")
		for key, value in output.items():
			summarize_output(value, f"{prefix}.{key}")
		return

	if isinstance(output, (list, tuple)):
		print(f"{prefix} type: {type(output).__name__}, length: {len(output)}")
		for i, value in enumerate(output):
			summarize_output(value, f"{prefix}[{i}]")
		return

	print(f"{prefix} python type: {type(output).__name__}")


def print_dataset_info(yolo_model):
	names = yolo_model.names if hasattr(yolo_model, "names") else {}
	if isinstance(names, dict):
		name_list = [names[k] for k in sorted(names.keys())]
	else:
		name_list = list(names)

	print("\n" + "=" * 80)
	print("Dataset / Label Type")
	print("-" * 80)
	print("Default YOLO11 pretrained weights use COCO label taxonomy.")
	print(f"number of classes: {len(name_list)}")
	print(f"class names: {name_list}")


def run_random_demo(yolo_model, imgsz):
	core_model = yolo_model.model
	core_model.eval()

	x = torch.rand(1, 3, imgsz, imgsz)
	print("\n" + "=" * 80)
	print("Random Input Demo")
	print("-" * 80)
	print(f"random input shape: {tuple(x.shape)}")

	with torch.no_grad():
		output = core_model(x)

	print("\nForward Output Summary")
	summarize_output(output)


def main():
	args = parse_args()

	try:
		import ultralytics
		from ultralytics import YOLO
	except Exception as err:
		print("Please install ultralytics first: pip install ultralytics")
		raise err

	yolo_model = YOLO(args.weights)

	print_header(torch.__version__, ultralytics.__version__, args.weights)
	print_model_info(yolo_model)
	run_random_demo(yolo_model, args.imgsz)
	print_dataset_info(yolo_model)


if __name__ == "__main__":
	main()