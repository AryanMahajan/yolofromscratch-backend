import torch
import cv2
import yaml
import numpy as np
from models.yolov1 import YOLOV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nms(boxes, scores, iou_threshold=0.5):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2]-box[0]) * (box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

def decode(pred, S=7, B=2, C=20, conf_thresh=0.2):
    pred = pred.view(S, S, B*5 + C).detach().cpu().numpy()
    boxes, scores, labels = [], [], []
    for i in range(S):
        for j in range(S):
            cell = pred[i, j]
            class_probs = cell[B*5:]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]
            for b in range(B):
                px, py, pw, ph, conf = cell[b*5:(b+1)*5]
                score = conf * class_score
                if score < conf_thresh:
                    continue
                cx = (j + px) / S
                cy = (i + py) / S
                w = pw**2
                h = ph**2
                x1 = (cx - w/2) * 448
                y1 = (cy - h/2) * 448
                x2 = (cx + w/2) * 448
                y2 = (cy + h/2) * 448
                boxes.append([x1,y1,x2,y2])
                scores.append(score)
                labels.append(class_id)
    if not boxes:
        return [], [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    keep = nms(boxes, scores, iou_threshold=0.5)
    return boxes[keep], scores[keep], labels[keep]

def run_one_image(img_path, config_path, ckpt_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = YOLOV1(
        im_size=cfg["dataset_params"]["im_size"],
        num_classes=cfg["dataset_params"]["num_classes"],
        model_config=cfg["model_params"]
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (448, 448))
    inp = torch.from_numpy(img_resized).permute(2,0,1).unsqueeze(0).float()/255.0
    inp = inp.to(device)

    with torch.no_grad():
        out = model(inp)

    boxes, scores, labels = decode(
        out[0],
        S=cfg["model_params"]["S"],
        B=cfg["model_params"]["B"],
        C=cfg["dataset_params"]["num_classes"],
        conf_thresh=cfg["train_params"]["infer_conf_threshold"]
    )

    class_map = {i: name for i, name in enumerate([
        'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
        'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'
    ])}

    if len(boxes) == 0:
        print("⚠️ No detections above threshold.")
    else:
        for box, score, cls in zip(boxes, scores, labels):
            x1,y1,x2,y2 = map(int, box)
            print(f"Class: {class_map[cls]}, Score: {score:.3f}, BBox: ({x1},{y1},{x2},{y2})")

if __name__ == "__main__":
    run_one_image("data/VOC2007/JPEGImages/000014.jpg", "config/voc.yaml", "voc/yolo_voc2007.pth")
