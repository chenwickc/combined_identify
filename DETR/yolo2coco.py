import os, json, yaml, cv2
from tqdm import tqdm

def yolo_to_coco(yaml_path, split, out_json):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    img_dir = data[split]  # e.g. ../train/images
    label_dir = img_dir.replace('images', 'labels')

    categories = []
    for k, v in data['names'].items():
        categories.append({"id": int(k), "name": v})

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    for img_name in tqdm(img_files, desc=f'Converting {split}'):
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": img_name,
            "height": h,
            "width": w
        })

        if os.path.exists(label_path):
            with open(label_path, 'r') as lf:
                for line in lf:
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    x = (xc - bw/2) * w
                    y = (yc - bh/2) * h
                    bw_pix = bw * w
                    bh_pix = bh * h

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls),
                        "bbox": [x, y, bw_pix, bh_pix],
                        "area": bw_pix * bh_pix,
                        "iscrowd": 0
                    })
                    ann_id += 1
        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    yaml_path = "dataset.yaml"
    yolo_to_coco(yaml_path, "train", "annotations/train.json")
    yolo_to_coco(yaml_path, "val", "annotations/val.json")
    yolo_to_coco(yaml_path, "test", "annotations/test.json")