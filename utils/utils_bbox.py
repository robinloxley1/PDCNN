import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

class BBoxUtility:
    def __init__(self, num_classes, rpn_pre_boxes=12000, rpn_nms=0.7, nms_iou=0.3, min_k=300):
        self.num_classes = num_classes
        self.rpn_pre_boxes = rpn_pre_boxes
        self._min_k = min_k

        # 非极大值抑制参数
        self.rpn_nms_threshold = rpn_nms
        self.classifier_nms_threshold = nms_iou

    @staticmethod
    def decode_boxes(mbox_loc, anchors, variances):
        """将回归框偏移量解码为实际边界框坐标"""
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]
        anchor_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        detections_center_x = mbox_loc[:, 0] * anchor_width * variances[0] + anchor_center_x
        detections_center_y = mbox_loc[:, 1] * anchor_height * variances[1] + anchor_center_y
        detections_width = np.exp(mbox_loc[:, 2] * variances[2]) * anchor_width
        detections_height = np.exp(mbox_loc[:, 3] * variances[3]) * anchor_height

        detections_xmin = detections_center_x - 0.5 * detections_width
        detections_ymin = detections_center_y - 0.5 * detections_height
        detections_xmax = detections_center_x + 0.5 * detections_width
        detections_ymax = detections_center_y + 0.5 * detections_height

        detections = np.stack([detections_xmin, detections_ymin, detections_xmax, detections_ymax], axis=-1)
        detections = np.clip(detections, 0.0, 1.0)
        return detections

    def detection_out_rpn(self, predictions, anchors, variances=[0.25, 0.25, 0.25, 0.25]):
        """RPN 层输出处理"""
        mbox_conf, mbox_loc = predictions
        results = []

        for i in range(len(mbox_loc)):
            detections = self.decode_boxes(mbox_loc[i], anchors, variances)
            c_confs = mbox_conf[i, :, 0]
            top_idx = np.argsort(c_confs)[::-1][:self.rpn_pre_boxes]

            boxes_to_process = detections[top_idx]
            scores_to_process = c_confs[top_idx]

            selected_idx = tf.image.non_max_suppression(
                boxes_to_process, scores_to_process, self._min_k, iou_threshold=self.rpn_nms_threshold
            ).numpy()

            results.append(boxes_to_process[selected_idx])
        return np.array(results)

    @staticmethod
    def frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape):
        """将预测框坐标转换回原图尺寸"""
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins = box_yx - box_hw / 2
        box_maxes = box_yx + box_hw / 2
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2],
                                box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def detection_out_classifier(self, predictions, rpn_results, image_shape, input_shape,
                                 confidence=0.3, variances=[0.125, 0.125, 0.25, 0.25], negative_use=False):
        """分类器层输出处理"""
        proposal_conf, proposal_loc = predictions
        results = []

        for i in range(len(proposal_conf)):
            detections = []

            rpn = rpn_results[i].copy()
            # 将 rpn 框从 (xmin, ymin, xmax, ymax) 转换成 (cx, cy, w, h)
            rpn[:, 2] = rpn[:, 2] - rpn[:, 0]
            rpn[:, 3] = rpn[:, 3] - rpn[:, 1]
            rpn[:, 0] = rpn[:, 0] + rpn[:, 2] / 2
            rpn[:, 1] = rpn[:, 1] + rpn[:, 3] / 2

            for j in range(proposal_conf[i].shape[0]):
                score = np.max(proposal_conf[i][j, :-1] if negative_use else proposal_conf[i][j])
                label = np.argmax(proposal_conf[i][j, :-1] if negative_use else proposal_conf[i][j])

                if score < confidence:
                    continue

                x, y, w, h = rpn[j]
                tx, ty, tw, th = proposal_loc[i][j, 4*label:4*(label+1)]

                cx = tx * variances[0] * w + x
                cy = ty * variances[1] * h + y
                bw = np.exp(tw * variances[2]) * w
                bh = np.exp(th * variances[3]) * h

                xmin = cx - bw / 2
                ymin = cy - bh / 2
                xmax = cx + bw / 2
                ymax = cy + bh / 2

                detections.append([xmin, ymin, xmax, ymax, score, label])

            if len(detections) > 0:
                detections = np.array(detections)
                box_xy = (detections[:, 0:2] + detections[:, 2:4]) / 2
                box_wh = detections[:, 2:4] - detections[:, 0:2]
                detections[:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)
                results.append(detections)
            else:
                results.append(np.array([]))

        return results
