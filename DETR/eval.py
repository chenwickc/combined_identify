import json, numpy as np

def compute_metrics(pred_file, gt_file, iou_thr=0.5):
    with open(pred_file, 'r') as f:
        preds = json.load(f)
    with open(gt_file, 'r') as f:
        gts = json.load(f)

    gt_by_img = {}
    for ann in gts['annotations']:
        gt_by_img.setdefault(ann['image_id'], []).append(ann)

    tp = fp = fn = 0
    for pred in preds:
        img_id = pred['image_id']
        gt_list = gt_by_img.get(img_id, [])
        matched = [False]*len(gt_list)
        ok = False
        for i, gt in enumerate(gt_list):
            if matched[i]: continue
            # 简单 IOU
            px,py,pw,ph = pred['bbox']
            gx,gy,gw,gh = gt['bbox']
            ix1=max(px,gx); iy1=max(py,gy)
            ix2=min(px+pw,gx+gw); iy2=min(py+ph,gy+gh)
            iw=max(ix2-ix1,0); ih=max(iy2-iy1,0)
            inter=iw*ih
            union=pw*ph+gw*gh-inter
            iou=inter/union if union>0 else 0

            if iou>=iou_thr and pred['category_id']==gt['category_id']:
                matched[i]=True
                ok=True
                break
        if ok:
            tp+=1
        else:
            fp+=1
    # FN = GT - TP
    fn = sum(len(v) for v in gt_by_img.values()) - tp

    precision = tp / (tp+fp+1e-9)
    recall = tp / (tp+fn+1e-9)
    acc = tp / (tp+fp+fn+1e-9)
    f1 = 2*precision*recall/(precision+recall+1e-9)

    return acc, recall, f1

if __name__ == '__main__':
    acc, recall, f1 = compute_metrics('work_dirs/preds.json', 'annotations/test.json')
    print(f"ACC={acc:.4f}, RECALL={recall:.4f}, F1={f1:.4f}")