import argparse
from pathlib import Path

from inference.AerialSegmentation import SegModel
from inference.inference_pl import run_inference, run_cascading_inference_on_file
from inference.scoring import score_predictions

FILE_NAME = Path(__file__).resolve()
INFERENCE_BASE_DIR = FILE_NAME.parent / "output"

def get_output_dirname(run_id: str) -> Path:
  output_dirname = INFERENCE_BASE_DIR / f"{run_id}"
  output_dirname.mkdir(parents=True, exist_ok=True)
  return output_dirname

def run_inference_on_test(args: argparse.Namespace):
  run_id = args.run_id
  dataset = args.dataset
  inference_type = args.inference_type
  inference_size = args.inference_size
  stride = args.stride
  smoothing = args.smoothing
  
  basedir = get_output_dirname(run_id)
  model = SegModel(run_id=run_id)
  run_inference(dataset, model, basedir, stride, smoothing, inference_size, inference_type)



if __name__ == "__main__":
  # TODO add argparser to choose which trained model to load 
  parser = argparse.ArgumentParser(description="Run inference on test set")
  parser.add_argument("--dataset", type=str, default="dataset-sample")
  parser.add_argument("--run_id", type=str, default="best_model")
  parser.add_argument("--inference_type", type=str, default=None)
  parser.add_argument("--inference_size", type=int, default=300)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--smoothing", action="store_true", default=False)

  args = parser.parse_args()

  run_inference_on_test(args)
