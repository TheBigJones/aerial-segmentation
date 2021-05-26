"""Script for the final submission. Retrain best model and run inference on test data"""
import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import wandb

import lit_models

from inference import run_inference, SegModel, score_predictions
from inference.run_inference_on_split import nullable_string, str2bool

# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)

PATIENCE=10


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    # Hide lines below until Lab 5
    parser.add_argument("--wandb", action="store_true", default=False)
    # Hide lines above until Lab 5
    parser.add_argument("--data_class", type=str, default="AerialData")
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--patience", type=int, default=PATIENCE)

    # Arguments for inference
    parser.add_argument("--inference_size", type=int, default=300)
    parser.add_argument("--inference_stride", type=int, default=300)
    parser.add_argument("--inference_split", nargs='+', default=["test"])
    parser.add_argument("--inference_type", type=nullable_string, default=None)
    parser.add_argument("--inference_batchsize", type=int, default=16)
    parser.add_argument("--highest_res_stage", type=int, default=0)
    parser.add_argument("--smoothing",type=str2bool, default=False)

    parser.add_argument("--elevation_alpha", type=float, default=0.0)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.
    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"data.{args.data_class}")
    data = data_class(args)
    model_class = _import_class(f"models.{args.model_class}")
    model = model_class(data_config=data.config(), args=args)

    patience = vars(args).get("patience", PATIENCE)

    if args.loss not in ("ctc", "transformer"):
        lit_model_class = lit_models.BaseLitModel

    if args.model_class == "Unet":
        lit_model_class = lit_models.UnetLitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(
            args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")

    if args.wandb:
        logger = pl.loggers.WandbLogger(project='aerialsegmenation-submission', entity='team_jf', settings=wandb.Settings(symlink=False))
        logger.watch(model)
        logger.log_hyperparams(vars(args))


    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=patience)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}", monitor="val_loss", mode="min"
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weights_summary = None  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, logger=logger, weights_save_path="training/logs")

    # If passing --auto_lr_find, this will set learning rate
    trainer.tune(lit_model, datamodule=data)

    trainer.fit(lit_model, datamodule=data)
    # pylint: enable=no-member
    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print("Best model saved at:", best_model_path)
        if args.wandb:
            wandb.save(best_model_path)
            print("Best model also uploaded to W&B")
    model = SegModel(checkpoint_path=best_model_path, model=model, args=args)

    dataset = args.dataset
    inference_size = args.inference_size
    inference_type = args.inference_type
    inference_stride = args.inference_stride
    smoothing = args.smoothing
    inference_split = args.inference_split
    inference_batchsize = args.inference_batchsize
    highest_res_stage = args.highest_res_stage
    basedir = wandb.run.dir


    print("---- Running Inference ----")
    run_inference(dataset, model=model, basedir=basedir, stride=inference_stride,
                  smoothing=smoothing, size=inference_size,
                  inference_type=inference_type, split=inference_split, batchsize=inference_batchsize
                  ,highest_res_stage=highest_res_stage)
    print("---- Scoring Predictions ----")
    score, _ = score_predictions(dataset, basedir=basedir, split=inference_split)
    wandb.config.update(score)
    wandb.summary.update(score)

if __name__ == "__main__":
    main()
