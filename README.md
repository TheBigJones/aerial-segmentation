# FSDL Project: AerialSegmentation

This is the GIT repository to our Full Stack Deep Learning Final Project. In the Project we worked on the benchmark given at https://wandb.ai/dronedeploy/dronedeploy-aerial-segmentation/benchmark. For a deeper dive on our project please read the [report](FSDL_Report.pdf).

## Installation

Run the setup bash script to install all dependencies.

```bash
bash setup.sh
```

Afterwads run the script to download and chip the dataset. This may take a while.

```
python download_data.py
```

## Usage
To train a model run the run_experiment.py script. The following parameters where used in our best run.

```
python training/run_experiment.py --wandb --batch_size=8 --data_class=AerialData --dataset=dataset-medium --elevation_alpha=10 --encoder_name=mobilenet_v2 --gpus=1 --loss=cross_entropy --lr=0.0001 --model_class=Unet --augmentations hflip vflip rotate
```

To run inference with a model use the run_inference_on_split.py script.

```
python training/run_inference_on_split.py --dataset=dataset-medium --run_id=9815f5bz --metric=val_f1 --project=aerialsegmenation-sweeps --size=4000 --smoothing=1 --stride=7 --split test --batchsize=1
```

With the run_id, project and metric flag one can determine the model used for inference. If the run_id flag is not set the best model is searched over all successfull runs in the project. 

## License
[MIT](https://choosealicense.com/licenses/mit/)