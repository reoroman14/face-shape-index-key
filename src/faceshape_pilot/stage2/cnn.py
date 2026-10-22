# CNN face shape classifier: model construction, deterministic setup,
# training loop and batched inference.

import os

import numpy as np

from faceshape_pilot import config

BACKBONES = ("efficientnet_b0", "resnet50")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Training hyperparameters, fixed before training and identical for both
# backbones.
EPOCHS = 20
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
# Training augmentation. It overlaps with the Step 3 perturbations (crop,
# rotation, colour), which makes the CNN more robust to them; any bias this
# introduces favours label stability.
AUG = {
    "random_resized_crop_scale": (0.85, 1.0),
    "random_resized_crop_ratio": (0.95, 1.05),
    "horizontal_flip_p": 0.5,
    "rotation_deg": 10.0,
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.1, "hue": 0.02},
}


def set_determinism(seed: int):
    import random

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(name: str, n_classes: int, pretrained: bool = True):
    import torch.nn as nn
    import torchvision.models as tvm

    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise ValueError(name)
    return m


def pretrained_weights_id(name: str) -> str:
    return {"efficientnet_b0": "torchvision EfficientNet_B0_Weights.IMAGENET1K_V1",
            "resnet50": "torchvision ResNet50_Weights.IMAGENET1K_V2"}[name]


def _train_transform():
    import torch
    from torchvision.transforms import v2

    return v2.Compose([
        v2.RandomResizedCrop(224, scale=AUG["random_resized_crop_scale"], ratio=AUG["random_resized_crop_ratio"],
                             antialias=True),
        v2.RandomHorizontalFlip(AUG["horizontal_flip_p"]),
        v2.RandomRotation(AUG["rotation_deg"]),
        v2.ColorJitter(**AUG["color_jitter"]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _to_input(batch_uint8, device):
    # (B, H, W, 3) uint8 numpy -> normalised (B, 3, H, W) float tensor.
    import torch

    x = torch.from_numpy(np.ascontiguousarray(batch_uint8)).to(device).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def predict_proba(model, crops_uint8: np.ndarray, batch: int = 128) -> np.ndarray:
    # Softmax probabilities for a stack of CROP_SIZE crops, no test-time
    # augmentation.
    import torch

    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops_uint8), batch):
            logits = model(_to_input(crops_uint8[i:i + batch], device))
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model_n_classes(model)))


def model_n_classes(model) -> int:
    import torch.nn as nn

    last = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    return last.out_features


def train(name: str, x_train, y_train, x_val, y_val, n_classes: int, seed: int, log=print):
    # Fine-tunes all layers. Returns (best_state_dict, history, best_epoch).
    import copy
    import time

    import torch
    import torch.nn as nn

    set_determinism(seed)
    device = torch.device("cuda")
    model = build_model(name, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    tf = _train_transform()
    gen = np.random.default_rng([seed, 7])

    history, best_acc, best_state, best_epoch = [], -1.0, None, None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        order = gen.permutation(len(x_train))
        tot_loss, tot_correct, n = 0.0, 0, 0
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            xb = torch.stack([tf(torch.from_numpy(x_train[j]).permute(2, 0, 1)) for j in idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).long().to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(idx)
            tot_correct += int((logits.argmax(1) == yb).sum())
            n += len(idx)
        sched.step()
        val_acc = float((predict_proba(model, x_val).argmax(1) == y_val).mean())
        secs = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tot_loss / n, "train_acc": tot_correct / n, "val_acc": val_acc,
                        "seconds": secs})
        log(f"[cnn:{name}] epoch {epoch:2d} loss {tot_loss / n:.4f} train_acc {tot_correct / n:.4f} "
            f"val_acc {val_acc:.4f} {secs:.0f}s", flush=True)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    return best_state, history, best_epoch


def load_trained(name: str, n_classes: int, state_path):
    import torch

    model = build_model(name, n_classes, pretrained=False)
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    return model.to("cuda").eval()


