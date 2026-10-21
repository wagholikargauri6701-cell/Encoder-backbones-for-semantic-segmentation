import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CamVidDataset, NUM_CLASSES, CAMVID_COLORS, IGNORE_INDEX, CLASS_NAMES_11
from model import get_model
from utils import compute_mean_iou

DATA_ROOT   = "/home/1d5f/.local/CV_and_AI/CamVid"
RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(PLOTS_DIR, exist_ok=True)

IMG_SIZE     = 256
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODERS     = ["vgg16", "resnet34"]
CLASS_NAMES  = CLASS_NAMES_11
SAMPLE_INDICES = [0, 10, 20, 30]

# ImageNet normalization for pretrained VGG16-BN and ResNet34 encoders
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD  = np.array([0.229, 0.224, 0.225])

def load_json(path):
    with open(path) as f:
        return json.load(f)

def build_class_colors():
    picked = {}
    for rgb, cls in CAMVID_COLORS.items():
        if cls != IGNORE_INDEX and cls not in picked:
            picked[cls] = rgb
    colors = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    for cls, rgb in picked.items():
        colors[cls] = rgb
    return colors

CLASS_COLORS = build_class_colors()

def class_map_to_rgb(class_map):
    h, w = class_map.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(len(CLASS_COLORS)):
        rgb[class_map == cls] = CLASS_COLORS[cls]
    rgb[class_map == 255] = 0
    return rgb

def unnormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * IMG_STD + IMG_MEAN, 0, 1)

def load_all_results():
    history, grad_norms, test_metrics, confusion, latency = {}, {}, {}, {}, {}
    for enc in ENCODERS:
        history[enc]      = load_json(os.path.join(RESULTS_DIR, f"history_{enc}.json"))
        grad_norms[enc]   = load_json(os.path.join(RESULTS_DIR, f"grad_norms_{enc}.json"))
        test_metrics[enc] = load_json(os.path.join(RESULTS_DIR, f"test_metrics_{enc}.json"))
        confusion[enc]    = np.array(load_json(os.path.join(RESULTS_DIR, f"confusion_matrix_{enc}.json")))
        latency[enc]      = load_json(os.path.join(RESULTS_DIR, f"latency_{enc}.json"))
    return history, grad_norms, test_metrics, confusion, latency

def load_trained_model(encoder_name):
    model = get_model(encoder_name=encoder_name, num_classes=NUM_CLASSES).to(DEVICE)
    ckpt  = os.path.join(CKPT_DIR, f"unet_{encoder_name}_best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model

#class imbalance of train and test 
def plot_class_imbalance():
    print("  computing pixel frequencies...")
    train_counts = compute_pixel_frequency("train")
    test_counts  = compute_pixel_frequency("test")

    with open(os.path.join(RESULTS_DIR, "class_pixel_frequencies.json"), "w") as f:
        json.dump({
            "train": train_counts.tolist(),
            "test": test_counts.tolist(),
            "class_names": CLASS_NAMES
        }, f, indent=2)

    train_pcts = train_counts / train_counts.sum() * 100.0
    test_pcts  = test_counts / test_counts.sum() * 100.0

    x = np.arange(NUM_CLASSES)
    width = 0.4

    plt.figure(figsize=(12, 6))

    bars1 = plt.bar(
        x - width/2, train_pcts, width,
        label="Train", color="steelblue"
    )

    bars2 = plt.bar(
        x + width/2, test_pcts, width,
        label="Test", color="orange"
    )

    # Add percentage labels
    for bars, pcts in [(bars1, train_pcts), (bars2, test_pcts)]:
        for bar, pct in zip(bars, pcts):
            if pct > 0.5:
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.2,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90
                )

    plt.xticks(x, CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    plt.ylabel("% of total pixels")
    plt.xlabel("Classes")
    plt.title("Class Distribution Comparison (Train vs Test)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOTS_DIR, "class_imbalance_train_vs_test.png"),
        dpi=150
    )
    plt.close()

#loss curves
def plot_loss_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for enc in ENCODERS:
        epochs = range(1, len(history[enc]["train_loss"]) + 1)
        axes[0].plot(epochs, history[enc]["train_loss"], label=f"{enc} train")
        axes[0].plot(epochs, history[enc]["val_loss"], linestyle="-", label=f"{enc} val")
        axes[1].plot(epochs, history[enc]["val_miou"], label=enc)
    axes[0].set(xlabel="epoch", ylabel="loss", title="train vs validation loss")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="epoch", ylabel="val mIoU", title="validation mIoU (convergence)")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "loss_and_convergence.png"), dpi=150)
    plt.close()


#training curves
def plot_training_dynamics(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for enc in ENCODERS:
        train_loss = np.array(history[enc]["train_loss"])
        val_loss   = np.array(history[enc]["val_loss"])
        epochs     = range(1, len(train_loss) + 1)

        axes[0].plot(epochs, val_loss - train_loss, label=enc)

        window = 5
        roll_std = [float(np.std(val_loss[max(0, i-window):i+1])) for i in range(len(val_loss))]
        axes[1].plot(epochs, roll_std, label=enc)

        if "epoch_time_s" in history[enc]:
            t = history[enc]["epoch_time_s"]
            axes[2].plot(range(1, len(t)+1), t, label=enc)

    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="-")
    axes[0].set(xlabel="epoch", ylabel="val_loss - train_loss", title="overfitting gap")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="epoch", ylabel="rolling std (window=5)", title="loss curve stability")
    axes[1].legend(fontsize=8)
    axes[2].set(xlabel="epoch", ylabel="seconds", title="training time per epoch")
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_dynamics.png"), dpi=150)
    plt.close()


#accuracy curves
def plot_accuracy_curve(history):
    plt.figure(figsize=(7, 4))
    for enc in ENCODERS:
        epochs = range(1, len(history[enc]["val_acc"]) + 1)
        plt.plot(epochs, history[enc]["val_acc"], label=enc)
    plt.xlabel("epoch"); plt.ylabel("pixel accuracy")
    plt.title("validation pixel accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_curve.png"), dpi=150)
    plt.close()

#learning curve
def plot_lr_curve(history):
    plt.figure(figsize=(7, 4))
    for enc in ENCODERS:
        epochs = range(1, len(history[enc]["lr"]) + 1)
        plt.plot(epochs, history[enc]["lr"], label=enc)
    plt.xlabel("epoch"); plt.ylabel("learning rate")
    plt.title("learning rate schedule"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "lr_curve.png"), dpi=150)
    plt.close()


#per class IOU
def plot_per_class_iou(test_metrics):
    x, width = np.arange(len(CLASS_NAMES)), 0.35
    plt.figure(figsize=(11, 5))
    for i, enc in enumerate(ENCODERS):
        vals = [test_metrics[enc]["per_class_iou"].get(str(c), 0.0) or 0.0
                for c in range(len(CLASS_NAMES))]
        bars = plt.bar(x + i * width, vals, width, label=enc)
        for bar, v in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    plt.xticks(x + width/2, CLASS_NAMES, rotation=45, ha="right")
    plt.ylabel("IoU"); plt.ylim(0, 1.05)
    plt.title("per-class IoU comparison"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "per_class_iou.png"), dpi=150)
    plt.close()

#confusion matrix
def plot_confusion_matrices(confusion):
    for enc in ENCODERS:
        cm      = confusion[enc]
        cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)
        plt.figure(figsize=(8, 7))
        plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90, fontsize=8)
        plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=8)
        plt.xlabel("predicted"); plt.ylabel("ground truth")
        plt.title(f"confusion matrix — UNet-{enc}")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                v = cm_norm[i, j]
                plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > 0.5 else "black", fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{enc}.png"), dpi=150)
        plt.close()

#efficiency vs accuracy  
def plot_efficiency_tradeoff(test_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = {"vgg16": "steelblue", "resnet34": "tomato"}

    for enc in ENCODERS:
        m       = test_metrics[enc]
        miou    = m["test_miou"]
        fps     = m.get("fps", 0.0)
        params  = m["total_params"] / 1e6
        gflops  = m.get("gflops", 0.0)
        size_mb = m["model_size_mb"]
        c       = colors[enc]

        for ax, x_val, x_label in [
            (axes[0], fps,    "FPS (higher = faster)"),
            (axes[1], params, "parameters (millions)"),
            (axes[2], gflops, "GFLOPs"),
        ]:
            ax.scatter(x_val, miou, s=size_mb*3, color=c, label=enc, alpha=0.85, zorder=3)
            ax.annotate(enc, (x_val, miou), textcoords="offset points", xytext=(5, 3), fontsize=9)
            ax.set_xlabel(x_label)
            ax.set_ylabel("test mIoU")
            ax.legend(fontsize=8)

    axes[0].set_title("accuracy vs inference speed")
    axes[1].set_title("accuracy vs model size")
    axes[2].set_title("accuracy vs compute cost")
    plt.suptitle("efficiency vs accuracy trade-off (bubble size = model MB)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "efficiency_tradeoff.png"), dpi=150)
    plt.close()

#per parameter memory
def plot_params_memory(test_metrics):
    x, width = np.arange(len(ENCODERS)), 0.35
    params  = [test_metrics[enc]["total_params"] / 1e6 for enc in ENCODERS]
    gpu_mem = [test_metrics[enc].get("gpu_peak_memory_mb", 0.0) for enc in ENCODERS]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    b1 = ax1.bar(x - width/2, params,  width, label="params (M)",    color="steelblue")
    ax1.set_ylabel("parameters (millions)")
    ax1.set_xticks(x); ax1.set_xticklabels(ENCODERS)

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + width/2, gpu_mem, width, label="GPU peak (MB)", color="tomato")
    ax2.set_ylabel("GPU peak memory (MB)")

    ax1.legend([b1, b2], [b1.get_label(), b2.get_label()], loc="upper left", fontsize=8)
    plt.title("parameters and GPU memory usage")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "params_and_gpu_memory.png"), dpi=150)
    plt.close()

#inference
def plot_inference_time_distribution(latency):
    data = [latency[enc] for enc in ENCODERS]
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, tick_labels=ENCODERS, patch_artist=True,
                boxprops=dict(facecolor="lightsteelblue"))
    plt.ylabel("inference time (ms)")
    plt.title("inference time distribution (50 runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "inference_time_boxplot.png"), dpi=150)
    plt.close()


#gradient distribution
def plot_gradient_distribution(grad_norms):
    plt.figure(figsize=(7, 4))
    for enc in ENCODERS:
        vals = np.array(grad_norms[enc], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            plt.hist(vals, bins=40, alpha=0.5, label=enc)
    plt.xlabel("gradient norm"); plt.ylabel("count")
    plt.title("gradient norm distribution across training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gradient_distribution.png"), dpi=150)
    plt.close()

#pixel frequency
def compute_pixel_frequency(split):
    dataset = CamVidDataset(DATA_ROOT, split=split, img_size=IMG_SIZE, augment=False)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    counts  = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, labels in loader:
        for cls in range(NUM_CLASSES):
            counts[cls] += int((labels.numpy() == cls).sum())
    return counts



def save_single_sample(idx, image_tensor, label_tensor, pred_vgg, pred_resnet):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, data, title in zip(
        axes,
        [unnormalize(image_tensor), class_map_to_rgb(label_tensor.numpy()),
         class_map_to_rgb(pred_vgg), class_map_to_rgb(pred_resnet)],
        ["input image", "ground truth", "UNet-VGG16", "UNet-ResNet34"],
    ):
        ax.imshow(data); ax.set_title(title, fontsize=10); ax.axis("off")

    patches = [mpatches.Patch(color=CLASS_COLORS[c]/255.0, label=CLASS_NAMES[c])
               for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=7,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f"sample index {idx}", fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(PLOTS_DIR, f"prediction_sample_{idx:04d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

def plot_prediction_samples():
    test_dataset  = CamVidDataset(DATA_ROOT, split="test", img_size=IMG_SIZE, augment=False)
    model_vgg     = load_trained_model("vgg16")
    model_resnet  = load_trained_model("resnet34")

    for idx in SAMPLE_INDICES:
        image, label = test_dataset[idx]
        inp = image.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_vgg    = model_vgg(inp).argmax(1).squeeze(0).cpu().numpy()
            pred_resnet = model_resnet(inp).argmax(1).squeeze(0).cpu().numpy()
        save_single_sample(idx, image, label, pred_vgg, pred_resnet)
        print(f"  saved prediction_sample_{idx:04d}.png")

def plot_failure_cases():
    test_dataset = CamVidDataset(DATA_ROOT, split="test", img_size=IMG_SIZE, augment=False)
    model_vgg    = load_trained_model("vgg16")
    model_resnet = load_trained_model("resnet34")

    scores = []
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, label = test_dataset[idx]
            inp = image.unsqueeze(0).to(DEVICE)
            pred_vgg    = model_vgg(inp).argmax(1).cpu()
            pred_resnet = model_resnet(inp).argmax(1).cpu()
            miou_v = compute_mean_iou(pred_vgg,    label.unsqueeze(0), NUM_CLASSES)
            miou_r = compute_mean_iou(pred_resnet, label.unsqueeze(0), NUM_CLASSES)
            scores.append((idx, (miou_v + miou_r) / 2.0, miou_v, miou_r))

    scores.sort(key=lambda x: x[1])
    worst = scores[:4]

    print("  worst 4 test samples by avg mIoU:")
    for idx, avg, v, r in worst:
        print(f"    idx={idx}  avg={avg:.3f}  vgg={v:.3f}  resnet={r:.3f}")

    for idx, avg, _, _ in worst:
        image, label = test_dataset[idx]
        inp = image.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_vgg    = model_vgg(inp).argmax(1).squeeze(0).cpu().numpy()
            pred_resnet = model_resnet(inp).argmax(1).squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, data, title in zip(
            axes,
            [unnormalize(image), class_map_to_rgb(label.numpy()),
             class_map_to_rgb(pred_vgg), class_map_to_rgb(pred_resnet)],
            ["input image", "ground truth", "UNet-VGG16", "UNet-ResNet34"],
        ):
            ax.imshow(data); ax.set_title(title, fontsize=10); ax.axis("off")

        plt.suptitle(f"failure case — sample {idx} (avg mIoU={avg:.3f})", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"failure_case_{idx:04d}.png"), dpi=150)
        plt.close()
        print(f"  saved failure_case_{idx:04d}.png")

def plot_per_class_boundary_f1(test_metrics):
    x, width = np.arange(len(CLASS_NAMES)), 0.35
    plt.figure(figsize=(11, 5))
    for i, enc in enumerate(ENCODERS):
        vals = [test_metrics[enc]["per_class_boundary_f1"].get(str(c), 0.0) or 0.0
                for c in range(len(CLASS_NAMES))]
        bars = plt.bar(x + i * width, vals, width, label=enc)
        for bar, v in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    plt.xticks(x + width/2, CLASS_NAMES, rotation=45, ha="right")
    plt.ylabel("Boundary F1"); plt.ylim(0, 1.05)
    plt.title("per-class boundary F1 comparison"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "per_class_boundary_f1.png"), dpi=150)
    plt.close()

def print_summary(test_metrics):
    print("\n" + "="*60)
    print(f"{'metric':<25} {'VGG16':>12} {'ResNet34':>12}")
    print("="*60)
    fields = [
        ("test mIoU",        "test_miou"),
        ("pixel accuracy",   "test_acc"),
        ("mean Dice",        "mean_dice"),
        ("macro precision",  "macro_precision"),
        ("macro recall",     "macro_recall"),
        ("mean boundary F1", "mean_boundary_f1"),
        ("params (M)",       None),
        ("model size (MB)",  "model_size_mb"),
        ("GFLOPs",           "gflops"),
        ("GPU memory (MB)",  "gpu_peak_memory_mb"),
        ("avg latency (ms)", "avg_latency_ms"),
        ("FPS",              "fps"),
    ]
    for label, key in fields:
        if key is None:
            vgg_val = test_metrics["vgg16"]["total_params"] / 1e6
            res_val = test_metrics["resnet34"]["total_params"] / 1e6
        else:
            vgg_val = test_metrics["vgg16"].get(key, 0.0) or 0.0
            res_val = test_metrics["resnet34"].get(key, 0.0) or 0.0
        print(f"{label:<25} {vgg_val:>12.4f} {res_val:>12.4f}")
    print("="*60)

def main():
    print("loading results..")
    history, grad_norms, test_metrics, confusion, latency = load_all_results()

    plot_class_imbalance()
    plot_loss_curves(history)
    plot_lr_curve(history)
    plot_accuracy_curve(history)
    plot_training_dynamics(history)
    plot_confusion_matrices(confusion)
    plot_per_class_iou(test_metrics)
    plot_efficiency_tradeoff(test_metrics)
    plot_params_memory(test_metrics)
    plot_inference_time_distribution(latency)
    plot_gradient_distribution(grad_norms)
    plot_prediction_samples()
    plot_failure_cases()
    plot_per_class_boundary_f1(test_metrics)
    print_summary(test_metrics)

    print(f"\nall plots saved to {PLOTS_DIR}/")

if __name__ == "__main__":
    main()



# IN BASH:
# python analysis.py
#     (run AFTER test.py - needs results/ folder to be get all the result records)
#mean, std = compute.py 