import os
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class DigitCocoDataset(Dataset):
    def __init__(self, json_path, image_dir, transforms=None):
        with open(json_path, "r") as f:
            coco = json.load(f)
        self.image_dir = image_dir
        self.transforms = transforms

        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = {}
        for ann in coco["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.images.keys())
        self.categories = {
            cat["id"]: cat["name"] for cat in coco["categories"]
        }

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.images[img_id]
        path = os.path.join(self.image_dir, info["file_name"])
        img = Image.open(path).convert("RGB")

        annos = self.annotations.get(img_id, [])
        boxes, labels = [], []
        for anno in annos:
            x, y, w, h = anno["bbox"]
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(anno["category_id"] - 1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_ids)


def get_model(num_classes=11):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def train_model():
    train_transform = T.Compose([T.ToTensor()])

    train_ds = DigitCocoDataset(
        '/kaggle/input/hw2-data/nycu-hw2-data/train.json',
        '/kaggle/input/hw2-data/nycu-hw2-data/train',
        train_transform
    )
    val_ds = DigitCocoDataset(
        '/kaggle/input/hw2-data/nycu-hw2-data/valid.json',
        '/kaggle/input/hw2-data/nycu-hw2-data/valid',
        T.ToTensor()
    )
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(11).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.0075,
        momentum=0.9, weight_decay=0.0005
    )

    for epoch in range(5):
        model.train()
        total_loss = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            imgs = [i.to(device) for i in imgs]
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]

            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.2f}")

    torch.save(model.state_dict(), "model_final.pth")
    print("Model saved to model_final.pth")
    return model


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(11)
    model.load_state_dict(torch.load("model_final.pth", map_location=device))
    model.to(device)
    model.eval()

    test_dir = '/kaggle/input/hw2-data/nycu-hw2-data/test'
    test_files = sorted(
        os.listdir(test_dir),
        key=lambda x: int(os.path.splitext(x)[0])
    )
    pred_json = []
    pred_csv = []

    for fname in tqdm(test_files, desc="Inferencing"):
        img_id = int(fname.split('.')[0])
        path = os.path.join(test_dir, fname)
        image = T.ToTensor()(Image.open(path).convert("RGB"))
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        digits = []
        for i in range(len(boxes)):
            if scores[i] < 0.7:
                continue
            x1, y1, x2, y2 = boxes[i]
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            pred_json.append({
                "image_id": img_id,
                "category_id": int(labels[i]) + 1,
                "bbox": [
                    float(x1), float(y1),
                    float(width), float(height)
                ],
                "score": float(scores[i]),
            })
            digit = labels[i] if labels[i] != 10 else 0
            digits.append((x1, digit))

        digits.sort(key=lambda x: x[0])
        pred_str = "".join(str(d) for _, d in digits)
        pred_str = pred_str if pred_str else "-1"
        pred_csv.append({
            "image_id": img_id,
            "pred_label": pred_str
        })

    with open("pred.json", "w") as f:
        json.dump(pred_json, f)

    pd.DataFrame(pred_csv).to_csv("pred.csv", index=False)
    print("Saved: pred.json & pred.csv")


if __name__ == "__main__":
    if not os.path.exists("model_final.pth"):
        train_model()
    run_inference()

