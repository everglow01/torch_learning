import argparse
import os
from urllib.parse import urlparse

import torch
from PIL import Image
from torchvision.models.segmentation import (
	DeepLabV3_MobileNet_V3_Large_Weights,
	deeplabv3_mobilenet_v3_large,
)


def load_model():
	weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
	try:
		model = deeplabv3_mobilenet_v3_large(weights=weights)
	except RuntimeError as err:
		if _is_corrupted_checkpoint_error(err):
			removed = _remove_corrupted_checkpoint(weights)
			if removed:
				print("Detected corrupted cached weights. Removed cache and retrying download once...")
			model = deeplabv3_mobilenet_v3_large(weights=weights)
		else:
			raise
	model.eval()
	return model, weights


def _is_corrupted_checkpoint_error(err):
	msg = str(err).lower()
	return "failed finding central directory" in msg or "pytorchstreamreader" in msg


def _remove_corrupted_checkpoint(weights):
	url_path = urlparse(weights.url).path
	filename = os.path.basename(url_path)
	if not filename:
		return False

	checkpoints_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
	checkpoint_path = os.path.join(checkpoints_dir, filename)
	if os.path.exists(checkpoint_path):
		os.remove(checkpoint_path)
		return True
	return False


def print_header_info(model, weights):
	print("=" * 80)
	print("Environment")
	print("-" * 80)
	print(f"torch: {torch.__version__}")
	print("image backend: Pillow (PIL)")

	print("\n" + "=" * 80)
	print("Model Overview")
	print("-" * 80)
	print(model)
	print("\nKey Modules")
	print(f"backbone: {type(model.backbone).__name__}")
	print(f"classifier: {type(model.classifier).__name__}")
	print(f"aux_classifier: {type(model.aux_classifier).__name__}")

	categories = weights.meta.get("categories", [])
	print("\n" + "=" * 80)
	print("Dataset / Label Type")
	print("-" * 80)
	print("Pretrained weights: COCO with VOC labels")
	print(f"Number of categories: {len(categories)}")
	print(f"Categories: {categories}")


def inspect_output(output_dict, title):
	print("\n" + "=" * 80)
	print(title)
	print("-" * 80)
	print(f"Output keys: {list(output_dict.keys())}")

	for key, value in output_dict.items():
		print(f"{key} shape: {tuple(value.shape)}")

	main_out = output_dict["out"]
	pred_mask = torch.argmax(main_out, dim=1)
	unique_ids = torch.unique(pred_mask).cpu().tolist()
	print(f"Pred mask shape: {tuple(pred_mask.shape)}")
	print(f"Unique class ids in mask: {unique_ids}")
	print(f"Unique class count: {len(unique_ids)}")


def run_random_demo(model):
	print("\n" + "=" * 80)
	print("Random Input Demo")
	print("-" * 80)
	x = torch.rand(1, 3, 520, 520)
	print(f"Random input shape: {tuple(x.shape)}")
	with torch.no_grad():
		output = model(x)
	inspect_output(output, "Random Input Output")


def run_image_demo(model, weights, image_path):
	if not os.path.exists(image_path):
		print("\n" + "=" * 80)
		print("Image Demo")
		print("-" * 80)
		print(f"Image path does not exist: {image_path}")
		return

	image_rgb = Image.open(image_path).convert("RGB")
	preprocess = weights.transforms()
	input_tensor = preprocess(image_rgb).unsqueeze(0)

	print("\n" + "=" * 80)
	print("Image Demo")
	print("-" * 80)
	print(f"Image path: {image_path}")
	print(f"Original image size (W, H): {image_rgb.size}")
	print(f"Model input tensor shape: {tuple(input_tensor.shape)}")

	with torch.no_grad():
		output = model(input_tensor)
	inspect_output(output, "Image Input Output")

	pred_mask = torch.argmax(output["out"], dim=1)[0]
	print(f"Mask tensor shape (H, W): {tuple(pred_mask.shape)}")


def parse_args():
	parser = argparse.ArgumentParser(
		description="Inspect DeepLabV3 with MobileNetV3-Large backbone"
	)
	parser.add_argument(
		"--image",
		type=str,
		default="",
		help="Optional image path for real-image inference demo",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	model, weights = load_model()
	print_header_info(model, weights)
	run_random_demo(model)

	if args.image:
		run_image_demo(model, weights, args.image)
	else:
		print("\n" + "=" * 80)
		print("Image Demo")
		print("-" * 80)
		print("No image path provided. Use --image <path> to run real-image inference.")


if __name__ == "__main__":
	main()