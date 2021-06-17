# FSDL Project: AerialSegmentation
<p float="center">
  <img src="presentation/images/2ef3a4994a_0CCD105428INSPIRE-ortho.png" width="45%" />
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="presentation/images/2ef3a4994a_0CCD105428INSPIRE-label.png" width="45%" /> 
</p>

This is the GIT repository to our Full Stack Deep Learning Final Project. In the Project we worked on the benchmark given at https://wandb.ai/dronedeploy/dronedeploy-aerial-segmentation/benchmark. 

In the report to our project we give an overview of the task we worked on, the given dataset and our approaches. Furthermore, we discuss the problems we have encountered during our work on the benchmark. We show, that one of the most limiting factors for improving on the given baseline model is the poor label quality of the given dataset. Given that the size of the orhomosaics in the dataset is considerably large, the images have to be chipped into smaller patches. When running inference on these patches, the context to be analyzed is lost. To tackle this issue, we used an approach we called Cascading Inference. For a deeper dive into our project, please read the [report](FSDL_Report.pdf).

## Installation
First, run the setup bash script to install all dependencies.

```bash
bash setup.sh
```

Afterwards run the download_data.py script to download and chip the dataset. This may take a while.

```
python download_data.py
```

Rename the config_template.dat file to config.dat and set the num_worker count ideal for your machine. When using the --read_config flag the script will automatically set the num_workers to the count defined in the config.dat file.

## Train
From the project folder run the run_experiment.py script. To replicate the model we submitted to the wandb benchmark, run the the script with the following arguments.

```
python training/run_experiment.py --wandb --batch_size=8 --data_class=AerialData --dataset=dataset-medium --elevation_alpha=10 --encoder_name=mobilenet_v2 --gpus=1 --loss=cross_entropy --lr=0.0001 --model_class=Unet --augmentations hflip vflip rotate
```

## Inference
To run inference use the run_inference_on_split.py script.

```
python training/run_inference_on_split.py --dataset=dataset-medium --run_id=9815f5bz --metric=val_f1 --project=aerialsegmenation-sweeps --size=4000 --smoothing=1 --stride=7 --split test --batchsize=1
```

With the run_id, project and metric flag one can determine the model used for inference. If the run_id flag is not set the best model is searched over all successful runs in the project.


## Submission
For our final submission we used the run_experiment_and_inference.py script.
If you would like to replicate our submission run the script with the following arguments.
```
training/run_experiment_and_inference.py --wandb --batch_size=8 --data_class=AerialData --dataset=dataset-medium --elevation_alpha=10 --encoder_name=mobilenet_v2 --gpus=1 --loss=cross_entropy --lr=0.0001 --model_class=Unet --augmentations hflip vflip rotate --read_config --inference_size=4000 --smoothing=1 --inference_stride=7 --inference_split test --inference_batchsize=1
```

Check out our submitted run by clicking the following link.<br/>
https://wandb.ai/team_jf/aerialsegmenation-submission/runs/1mygfkxh/overview?workspace=user-
## License
[MIT](https://choosealicense.com/licenses/mit/)
