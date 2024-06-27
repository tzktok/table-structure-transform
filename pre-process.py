#nms

class Preprocess:
    def __init__(self, iou_threshold=0.4):
        self.iou_threshold = iou_threshold

    def apply_nms(self, boxes):
        """Apply Non-Maximum Suppression to given boxes."""
        boxes.sort(key=lambda x: x[0], reverse=True)  # Sort by confidence descending
        keep = []
        while boxes:
            max_confidence_box = boxes.pop(0)
            keep.append(max_confidence_box)
            boxes = [box for box in boxes if self.calculate_iou(max_confidence_box[1], box[1]) < self.iou_threshold]
        return keep

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, x2, y2 = box1
        xx1, yy1, xx2, yy2 = box2

        inter_area = max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (xx2 - xx1) * (yy2 - yy1)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou
