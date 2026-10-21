import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -- Loss Functions --

class DiceLoss(nn.Module):
    #Multi-class soft Dice loss. Void pixels (ignore_index) are excluded.

    def __init__(self, num_classes=11, smooth=1.0, ignore_index=255):
        super().__init__()
        self.num_classes   = num_classes
        self.smooth        = smooth
        self.ignore_index  = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        valid    = targets != self.ignore_index
        targets_c = targets.clone()
        targets_c[~valid] = 0

        one_hot = (
            F.one_hot(targets_c, self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        valid   = valid.unsqueeze(1)
        probs   = probs   * valid
        one_hot = one_hot * valid

        dims  = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        dice  = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    #Weighted sum of CrossEntropy and Dice loss.

    def __init__(self, ce_weight=0.5, dice_weight=0.5, num_classes=11, ignore_index=255):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)

# -- Segmentation Metrics ---
@torch.no_grad()
def compute_per_class_iou(preds, labels, num_classes, ignore_index=255):
    preds  = preds.cpu()
    labels = labels.cpu()
    valid  = labels != ignore_index
    result = {}
    for cls in range(num_classes):
        pred_c = (preds == cls) & valid
        gt_c   = (labels == cls) & valid
        inter  = (pred_c & gt_c).sum().item()
        union  = (pred_c | gt_c).sum().item()
        result[cls] = inter / union if union > 0 else float("nan")
    return result

@torch.no_grad()
def compute_mean_iou(preds, labels, num_classes, ignore_index=255):
    per_class = compute_per_class_iou(preds, labels, num_classes, ignore_index)
    vals = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0

@torch.no_grad()
def compute_pixel_accuracy(preds, labels, ignore_index=255):
    preds  = preds.cpu()
    labels = labels.cpu()
    valid   = labels != ignore_index
    correct = ((preds == labels) & valid).sum().item()
    total   = valid.sum().item()
    return correct / total if total > 0 else 0.0

@torch.no_grad()
def compute_per_class_dice(preds, labels, num_classes, ignore_index=255):
    preds  = preds.cpu()
    labels = labels.cpu()
    valid  = labels != ignore_index
    result = {}
    for cls in range(num_classes):
        pred_c = (preds == cls) & valid
        gt_c   = (labels == cls) & valid
        inter  = (pred_c & gt_c).sum().item()
        total  = pred_c.sum().item() + gt_c.sum().item()
        result[cls] = (2.0 * inter) / total if total > 0 else float("nan")
    return result

@torch.no_grad()
def compute_mean_dice(preds, labels, num_classes, ignore_index=255):
    per_class = compute_per_class_dice(preds, labels, num_classes, ignore_index)
    vals = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0

@torch.no_grad()
def compute_precision_recall(preds, labels, num_classes, ignore_index=255):
    preds  = preds.cpu()
    labels = labels.cpu()
    valid  = labels != ignore_index
    precision, recall = {}, {}

    for cls in range(num_classes):
        pred_c = (preds == cls) & valid
        gt_c   = (labels == cls) & valid
        tp = (pred_c &  gt_c).sum().item()
        fp = (pred_c & ~gt_c).sum().item()
        fn = (~pred_c & gt_c).sum().item()
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall[cls]    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    valid_p = [v for v in precision.values() if not np.isnan(v)]
    valid_r = [v for v in recall.values()    if not np.isnan(v)]
    macro_p = float(np.mean(valid_p)) if valid_p else 0.0
    macro_r = float(np.mean(valid_r)) if valid_r else 0.0
    return precision, recall, macro_p, macro_r

@torch.no_grad()
def compute_confusion_matrix(preds, labels, num_classes, ignore_index=255):
    preds  = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    valid  = labels != ignore_index
    preds  = preds[valid]
    labels = labels[valid]
    cm = np.bincount(
        num_classes * labels + preds,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return cm.astype(np.int64)

# -- Model Complexity ---

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model):
    total = sum(p.numel() for p in model.parameters())
    return total * 4 / (1024 ** 2)   # float32 = 4 bytes

def compute_flops(model, input_size=(1, 3, 256, 256), device="cpu"):
    try:
        from torch.profiler import profile, ProfilerActivity
        x = torch.randn(*input_size).to(device)
        model.eval()
        with profile(activities=[ProfilerActivity.CPU],
                     record_shapes=True, with_flops=True) as prof:
            with torch.no_grad():
                model(x)
        total = sum(e.flops for e in prof.key_averages() if e.flops is not None)
        return total / 1e9
    except Exception:
        # fallback: rough estimate from parameter count
        return count_parameters(model) * 2.0 / 1e9

# -- Inference Latency --

def measure_inference_latency(model, device, input_size=(1, 3, 256, 256), n_runs=50, warmup=10):
    model.eval()
    x = torch.randn(*input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
        "max_ms":  float(np.max(times)),
        "fps":     float(1000.0 / np.mean(times)),
        "all_ms":  times.tolist(),
    }

# -- Boundary F1 --

def compute_boundary_f1(preds, labels, num_classes, ignore_index=255):
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:
        print("scipy not installed — boundary F1 skipped")
        return {cls: 0.0 for cls in range(num_classes)}

    preds     = preds.cpu().numpy()
    labels    = labels.cpu().numpy()
    valid     = labels != ignore_index
    structure = np.ones((3, 3), dtype=bool)
    result    = {}

    for cls in range(num_classes):
        tp = fp = fn = 0
        pred_c = (preds == cls) & valid
        gt_c   = (labels == cls) & valid

        for i in range(preds.shape[0]):
            pred_b = pred_c[i] ^ binary_dilation(pred_c[i], structure)
            gt_b   = gt_c[i]   ^ binary_dilation(gt_c[i],   structure)
            pred_d = binary_dilation(pred_b, structure)
            gt_d   = binary_dilation(gt_b,   structure)
            tp += (pred_b &  gt_d).sum()
            fp += (pred_b & ~gt_d).sum()
            fn += (gt_b   & ~pred_d).sum()

        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        result[cls] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return result