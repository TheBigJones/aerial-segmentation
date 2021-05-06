import argparse
from pathlib import Path
import wandb

from inference.AerialSegmentation import SegModel
from inference.inference_pl import run_inference, run_cascading_inference_on_file
from inference.scoring import score_predictions

FILE_NAME = Path(__file__).resolve()
INFERENCE_BASE_DIR = FILE_NAME.parent

def get_output_dirname(run_id: str) -> Path:
  output_dirname = INFERENCE_BASE_DIR / f"{run_id}"
  output_dirname.mkdir(parents=True, exist_ok=True)
  return output_dirname

def run_inference_on_test(args: argparse.Namespace):
  training_run_id = args.training_run_id
  dataset = args.dataset
  inference_type = args.inference_type
  inference_size = args.inference_size
  stride = args.stride
  smoothing = args.smoothing
  split = args.split
  basedir = wandb.run.dir
  model = SegModel(run_id=training_run_id)
  run_inference(dataset, model, basedir, stride, smoothing, inference_size, inference_type, split)
  score, _ = score_predictions(dataset, basedir=basedir)
  wandb.config.update(score)
  wandb.summary.update(score)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Run inference on test set")
  parser.add_argument("--dataset", type=str, default="dataset-sample")
  parser.add_argument("--split", type=str, default="val")
  parser.add_argument("--run_id", type=str, default="best_model")
  parser.add_argument("--inference_type", type=str, default=None)
  parser.add_argument("--inference_size", type=int, default=300)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--smoothing", action="store_true", default=False)

  args = parser.parse_args()
  config_update = vars(args)
  config_update["training_run_id"] = config_update.pop("run_id")
  wandb.init(project="aerialsegmentation-inference", entity="team_jf", dir=INFERENCE_BASE_DIR)
  wandb.summary.update(config_update)
  run_inference_on_test(args)
