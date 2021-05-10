import argparse
from pathlib import Path
import wandb

from inference.AerialSegmentation import SegModel
from inference.inference_pl import run_inference
from inference.scoring import score_predictions
from training.save_model import save_model

FILE_NAME = Path(__file__).resolve()
INFERENCE_BASE_DIR = FILE_NAME.parent

def nullable_string(val):
    if not val:
        return None
    return val

def str2bool(v):
  """https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
  if isinstance(v, bool):
      return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
  else:
      raise argparse.ArgumentTypeError('Boolean value expected.')

def get_output_dirname(run_id: str) -> Path:
  output_dirname = INFERENCE_BASE_DIR / f"{run_id}"
  output_dirname.mkdir(parents=True, exist_ok=True)
  return output_dirname

def run_inference_on_test(args: argparse.Namespace):
  training_run_id = args.training_run_id
  dataset = args.dataset
  inference_type = args.inference_type
  size = args.size
  stride = args.stride
  smoothing = args.smoothing
  split = args.split
  batchsize = args.batchsize
  basedir = wandb.run.dir
  model = SegModel(run_id=training_run_id)
  print("---- Running Inference ----")
  run_inference(dataset, model=model, basedir=basedir, stride=stride,
                smoothing=smoothing, size=size,
                inference_type=inference_type, split=split, batchsize=batchsize)
  print("---- Scoring Predictions ----")
  score, _ = score_predictions(dataset, basedir=basedir, split=split)
  wandb.config.update(score)
  wandb.summary.update(score)

if __name__ == "__main__":
  # setup parser
  parser = argparse.ArgumentParser(description="Run inference on test set")

  # inference args
  parser.add_argument("--dataset", type=str, default="dataset-sample")
  parser.add_argument("--split", nargs='+', default=["val"])
  parser.add_argument("--run_id", type=str, default="best_model")
  parser.add_argument("--inference_type", type=nullable_string, default=None)
  parser.add_argument("--size", type=int, default=300)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--batchsize", type=int, default=16)
  parser.add_argument("--smoothing",type=str2bool, default=False)

  # save_model args
  parser.add_argument("--entity", type=str, default="team_jf")
  parser.add_argument("--project", type=str, default="aerialsegmenation")
  parser.add_argument("--trained_data_class", type=str, default="AerialData")
  parser.add_argument("--metric", type=str, default="f1_mean")
  parser.add_argument("--mode", type=str, default="max")

  args = parser.parse_args()

  # save the model from the given run_id to the artifacts folder
  save_model(args)

  # init wandb run and update summary
  config_update = vars(args)
  config_update["training_run_id"] = config_update.pop("run_id")
  wandb.init(project="aerialsegmentation-inference", entity="team_jf", dir=INFERENCE_BASE_DIR)
  wandb.summary.update(config_update)
  run_inference_on_test(args)
