import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CamVidDataset, NUM_CLASSES
from model import get_model
from utils import (
    compute_confusion_matrix,
    compute_mean_dice,
    compute_mean_iou,
    compute_per_class_dice,
    compute_per_class_iou,
    compute_pixel_accuracy,
    compute_precision_recall,
    compute_boundary_f1,
    count_parameters,
    measure_inference_latency,
    model_size_mb,
    compute_flops,
    set_seed,
)

DATA_ROOT   = "/home/1d5f/.local/CV_and_AI/CamVid"
RESULTS_DIR = "results"
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE    = 256
NUM_WORKERS = 2
SEED        = 42

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODERS = ["vgg16", "resnet34"]

def load_model(encoder_name):
    model = get_model(encoder_name=encoder_name, num_classes=NUM_CLASSES).to(DEVICE)
    ckpt  = os.path.join(CKPT_DIR, f"unet_{encoder_name}_best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model

def measure_gpu_memory(model, device, input_size=(1, 3, 256, 256)):
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(*input_size).to(device)
    with torch.no_grad():
        model(x)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)

def evaluate(encoder_name):
    print(f"\nEvaluating UNet-{encoder_name}")
    set_seed(SEED)

    test_dataset = CamVidDataset(DATA_ROOT, split="test", img_size=IMG_SIZE, augment=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

    model = load_model(encoder_name)

    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            preds_all.append(model(images).argmax(1).cpu())
            labels_all.append(labels)

    preds_all  = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)

    miou           = compute_mean_iou(preds_all, labels_all, NUM_CLASSES)
    acc            = compute_pixel_accuracy(preds_all, labels_all)
    mean_dice      = compute_mean_dice(preds_all, labels_all, NUM_CLASSES)
    per_class_iou  = compute_per_class_iou(preds_all, labels_all, NUM_CLASSES)
    per_class_dice = compute_per_class_dice(preds_all, labels_all, NUM_CLASSES)
    per_class_prec, per_class_rec, macro_prec, macro_rec = compute_precision_recall(
        preds_all, labels_all, NUM_CLASSES
    )
    cm                  = compute_confusion_matrix(preds_all, labels_all, NUM_CLASSES)
    per_class_bf1       = compute_boundary_f1(preds_all, labels_all, NUM_CLASSES)

    total_params  = count_parameters(model)
    size_mb       = model_size_mb(model)
    gflops        = compute_flops(model, input_size=(1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
    gpu_peak_mb   = measure_gpu_memory(model, DEVICE)
    latency_info  = measure_inference_latency(model, DEVICE, n_runs=50, warmup=10)

    bf1_vals = [v for v in per_class_bf1.values() if not np.isnan(v)]
    mean_bf1 = float(np.mean(bf1_vals)) if bf1_vals else 0.0

    print(f"  test mIoU        : {miou:.4f}")
    print(f"  pixel accuracy   : {acc:.4f}")
    print(f"  mean Dice        : {mean_dice:.4f}")
    print(f"  macro precision  : {macro_prec:.4f}")
    print(f"  macro recall     : {macro_rec:.4f}")
    print(f"  mean boundary F1 : {mean_bf1:.4f}")
    print(f"  parameters       : {total_params/1e6:.2f}M")
    print(f"  model size       : {size_mb:.1f} MB")
    print(f"  GFLOPs           : {gflops:.2f}")
    print(f"  GPU peak memory  : {gpu_peak_mb:.1f} MB")
    print(f"  avg latency      : {latency_info['mean_ms']:.2f} ms")
    print(f"  FPS              : {latency_info['fps']:.1f}")

    metrics = {
        "test_miou":            miou,
        "test_acc":             acc,
        "mean_dice":            mean_dice,
        "macro_precision":      macro_prec,
        "macro_recall":         macro_rec,
        "mean_boundary_f1":     mean_bf1,
        "per_class_iou":        {str(k): v for k, v in per_class_iou.items()},
        "per_class_dice":       {str(k): v for k, v in per_class_dice.items()},
        "per_class_precision":  {str(k): v for k, v in per_class_prec.items()},
        "per_class_recall":     {str(k): v for k, v in per_class_rec.items()},
        "per_class_boundary_f1":{str(k): v for k, v in per_class_bf1.items()},
        "total_params":         total_params,
        "model_size_mb":        size_mb,
        "gflops":               gflops,
        "gpu_peak_memory_mb":   gpu_peak_mb,
        "avg_latency_ms":       latency_info["mean_ms"],
        "fps":                  latency_info["fps"],
    }

    with open(os.path.join(RESULTS_DIR, f"test_metrics_{encoder_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(RESULTS_DIR, f"confusion_matrix_{encoder_name}.json"), "w") as f:
        json.dump(cm.tolist(), f)
    with open(os.path.join(RESULTS_DIR, f"latency_{encoder_name}.json"), "w") as f:
        json.dump(latency_info["all_ms"], f)

    gpu_mem_info = {"gpu_peak_memory_mb": gpu_peak_mb, "device": str(DEVICE)}
    if DEVICE.type == "cuda":
        gpu_mem_info["gpu_name"] = torch.cuda.get_device_name(0)
    with open(os.path.join(RESULTS_DIR, f"gpu_memory_{encoder_name}.json"), "w") as f:
        json.dump(gpu_mem_info, f, indent=2)

    print(f"  results saved to {RESULTS_DIR}/")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder", type=str, default="both",
                        choices=["vgg16", "resnet34", "both"])
    args     = parser.parse_args()
    encoders = ["vgg16", "resnet34"] if args.encoder == "both" else [args.encoder]
    for enc in encoders:
        evaluate(enc)

if __name__ == "__main__":
    main()



# IN BASH:
# python test.py \
#     --encoder both \
#     --data_root "/home/1d5f/.local/CV_and_AI/CamVid" \
#     --img_size 256 \
#     --num_classes 11 \
#     --batch_size 1 \
#     --num_workers 2
#
# python test.py --encoder vgg16
# python test.py --encoder resnet34

