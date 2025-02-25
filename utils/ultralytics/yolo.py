import numpy as np
import cv2

def preprocess(im, imgsz, dtype):
    im = np.stack(pre_transform(im, imgsz))
    im = im[..., ::-1]
    im = np.ascontiguousarray(im).astype(dtype)
    im /= 255.0
    return im
        
def pre_transform(im, imgsz):
    return [cv2.resize(im[0], imgsz, interpolation=cv2.INTER_LINEAR) for x in im]

def postprocess(preds, imgsz, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300, nc=0, max_time_img=0.05, max_nms=30000, max_wh=7680, in_place=True, rotated=False):
    preds[:, [0, 2]] *= imgsz[0] ; preds[:, [1, 3]] *= imgsz[1]
    xc = np.max(preds[:, 4: nc + 4], axis = 1) > conf_thres
    preds = np.transpose(preds, (0, 2, 1)) 
    preds[..., :4] = xywh2xyxy(preds[..., :4])
    x = preds[0][xc[0]]

    if not x.shape[0]:
        return None
    box, cls, keypoints = x[:, :4], x[:, 4:5], x[:, 5:]
    j = np.argmax(cls, axis=1)
    conf = cls[[i for i in range(len(j))], j]
    concatenated = np.concatenate((box, conf.reshape(-1, 1), j.reshape(-1, 1).astype(float), keypoints), axis=1)
    x = concatenated[conf.flatten() > conf_thres]

    if x.shape[0] > max_nms:  # excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
    cls = x[:, 5:6] * (0 if agnostic else max_wh)
    scores, boxes = x[:, 4], x[:, :4] + cls
    i = non_max_suppression(boxes, scores, iou_thres)
    return [x[i[:max_det]]]
        
class LetterBox:
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

def visualizer(image, results, labels):
    for bboxes in results:
      x1, y1, x2, y2 = int(bboxes[0] * image.shape[1] / 640), int(bboxes[1] * image.shape[0] / 640), int(bboxes[2] * image.shape[1] / 640), int(bboxes[3] * image.shape[0] / 640)
      conf, cls = bboxes[4] , bboxes[5]
      cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
      cv2.putText(image, f'{labels[int(cls)]} {conf:.2f}', (x1, y1 - 2), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
    return image

def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y

def non_max_suppression(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)
