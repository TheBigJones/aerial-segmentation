"""
Module to find and save the best model trained on a given dataset to artifacts directory.
Run:
python training/save_best_model.py --entity=fsdl-user \
                                   --project=PL_FRAMEWORK \
                                   --trained_data_class=Baufi
To find entity and project, open any wandb run in web browser and look for the field "Run path" in "Overview" page.
"Run path" is of the format "<entity>/<project>/<run_id>".
"""
import argparse
import sys
import shutil
import json
from pathlib import Path
import tempfile
from typing import Optional, Union
import wandb


FILE_NAME = Path(__file__).resolve()
ARTIFACTS_BASE_DIRNAME = FILE_NAME.parents[1] / "artifacts"
TRAINING_LOGS_DIRNAME = FILE_NAME.parent / "logs"


def save_best_model():
    """Find and save the best model trained on a given dataset to artifacts directory."""
    parser = _setup_parser()
    args = parser.parse_args()

    if args.mode == "min":
        default_metric_value = sys.maxsize
        sort_reverse = False
    else:
        default_metric_value = 0
        sort_reverse = True

    api = wandb.Api()
    if args.run_id is None:
      runs = api.runs(f"{args.entity}/{args.project}")  # , filters={"config.data_class": args.trained_data_class})
      sorted_runs = sorted(
          runs,
          key=lambda run: _get_summary_value(wandb_run=run, key=args.metric, default=default_metric_value),
          reverse=sort_reverse,
      )

      run = sorted_runs[0]
      summary = run.summary
      print(f"Best run ({run.name}, {run.id}) picked from {len(runs)} runs with the following metrics:")
      print(f" - val_loss: {summary['val_loss']}, val_acc: {summary['val_acc']}, val_f1: {summary['val_f1']}")
    else:
      run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
      summary = run.summary
      print(f"Run ({run.name}, {run.id}) with the following metrics:")
      print(f" - val_loss: {summary['val_loss']}, val_acc: {summary['val_acc']}, val_f1: {summary['val_f1']}")
    artifacts_dirname = _get_artifacts_dirname(args.trained_data_class, args.run_id)
    with open(artifacts_dirname / "config.json", "w") as file:
        json.dump(run.config, file, indent=4)
    with open(artifacts_dirname / "run_command.txt", "w") as file:
        file.write(_get_run_command(run))
    _save_model_weights(wandb_run=run, project=args.project, output_dirname=artifacts_dirname)


def _get_artifacts_dirname(trained_data_class: str, run_id: str) -> Path:
    """Return artifacts dirname."""
    if run_id == None:
      run_id = "best_model"
    for keyword in ["aerialdata"]:
      if keyword in trained_data_class.lower():
          artifacts_dirname = ARTIFACTS_BASE_DIRNAME / f"{keyword}_{run_id}"
          artifacts_dirname.mkdir(parents=True, exist_ok=True)
          break
    return artifacts_dirname


def _save_model_weights(wandb_run: wandb.apis.public.Run, project: str, output_dirname: Path):
    """Save checkpointed model weights in output_dirname."""
    weights_filename = _copy_local_model_checkpoint(run_id=wandb_run.id, project=project, output_dirname=output_dirname)
    if weights_filename is None:
        weights_filename = _download_model_checkpoint(wandb_run, output_dirname)
        assert weights_filename is not None, "Model checkpoint not found"


def _copy_local_model_checkpoint(run_id: str, project: str, output_dirname: Path) -> Optional[Path]:
    """Copy model checkpoint file on system to output_dirname."""
    checkpoint_filenames = list((TRAINING_LOGS_DIRNAME / project / run_id).glob("**/*.ckpt"))
    if not checkpoint_filenames:
        return None
    shutil.copyfile(src=checkpoint_filenames[0], dst=output_dirname / "model.pt")
    print(f"Model checkpoint found on system at {checkpoint_filenames[0]}")
    return checkpoint_filenames[0]


def _download_model_checkpoint(wandb_run: wandb.apis.public.Run, output_dirname: Path) -> Optional[Path]:
    """Download model checkpoint to output_dirname."""
    checkpoint_wandb_files = [file for file in wandb_run.files() if file.name.endswith(".ckpt")]
    if not checkpoint_wandb_files:
        return None

    wandb_file = checkpoint_wandb_files[0]
    with tempfile.TemporaryDirectory() as tmp_dirname:
        wandb_file.download(root=tmp_dirname, replace=True)
        checkpoint_filename = f"{tmp_dirname}/{wandb_file.name}"
        shutil.copyfile(src=checkpoint_filename, dst=output_dirname / "model.pt")
        print("Model checkpoint downloaded from wandb")
    return output_dirname / "model.pt"


def _get_run_command(wandb_run: wandb.apis.public.Run) -> str:
    """Return python run command for input wandb_run."""
    with tempfile.TemporaryDirectory() as tmp_dirname:
        wandb_file = wandb_run.file("wandb-metadata.json")
        with wandb_file.download(root=tmp_dirname, replace=True) as file:
            metadata = json.load(file)

    return f"python {metadata['program']} " + " ".join(metadata["args"])


def _get_summary_value(wandb_run: wandb.apis.public.Run, key: str, default: int) -> Union[int, float]:
    """Return numeric value at summary[key] for wandb_run if it is valid, else return default."""
    value = wandb_run.summary.get(key, default)
    if not isinstance(value, (int, float)):
        value = default
    return value


def _setup_parser() -> argparse.ArgumentParser:
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--entity", type=str, default="team_jf")
    parser.add_argument("--project", type=str, default="aerialsegmenation")
    parser.add_argument("--trained_data_class", type=str, default="AerialData")
    parser.add_argument("--metric", type=str, default="f1_mean")
    parser.add_argument("--mode", type=str, default="max")

    parser.add_argument("--run_id", type=str, default=None)
    return parser


if __name__ == "__main__":
    save_best_model()
