from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode

for d in ["train", "test"]:
    register_coco_instances(f"laboro_tomato_{d}", {},
                            f"../content/laboro_tomato/annotations/{d}.json",
                            f"../content/laboro-tomato/{d}")
# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
register_coco_instances("tomato_train", {}, "/content/laboro_tomato/annotations/train.json", "/content/laboro_tomato/train")
register_coco_instances("tomato_val", {}, "/content/laboro_tomato/annotations/test.json", "/content/laboro_tomato/test")
