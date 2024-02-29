from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer

with open('cfg.pkl', 'rb') as f:
    cfg = pickle.load(f)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

evaluator = COCOEvaluator("tomato_val", False, output_dir="./output/")
test_loader = build_detection_test_loader(cfg, "tomato_val")
inference_on_dataset(trainer.model, test_loader, evaluator)
