# Code Citations

## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
    loss_obj
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
    loss_obj = (self.bcew
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
    loss_obj = (self.bcewithlog_loss(obj_preds.
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
    loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
    loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
    loss_cls = (
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
    loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
    loss_cls = (self.bcewithlog_loss(cls_
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
    loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
    loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
    loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
    loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num
```


## License: unknown
https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/detector/yolox/yolox/models/yolo_head.py

```
.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
    loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
    loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
```

