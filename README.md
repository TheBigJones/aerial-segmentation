# FSDL Project: AerialSegmentation
<p float="center">
  <img src="presentation/images/2ef3a4994a_0CCD105428INSPIRE-ortho.png" width="45%" />
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="presentation/images/2ef3a4994a_0CCD105428INSPIRE-label.png" width="45%" /> 
</p>

This is the GIT repository to our Full Stack Deep Learning Final Project. In the Project we worked on the benchmark given at https://wandb.ai/dronedeploy/dronedeploy-aerial-segmentation/benchmark. 

In the report to our project we give an overview of the task we worked on, the given dataset and our approaches. Furthermore, we discuss the problems we have encountered during our work on the benchmark. We show, that one of the most limiting factors for improving on the given baseline model is the poor label quality of the given dataset. Given that the size of the orhomosaics in the dataset are rather large, the images have to be chipped into smaller patches in order to be processable. However, when running inference on these patches, the context, i.e. the surroundings of the patch, is lost, which limits the accuracy of the predictions. One way to reduce this decrease in performance is to use larger patches, which is very much possible in principle, especially given that we chose fully-convolutional networks as models, which can deal with inputs of different sizes (with certain limitations). Having said that, in practice there are limitations due to memory constraints, which make this approach only feasible to a certain point. To improve on this, we used an approach we call Cascading Inference, which relies on predicting on the same picture in different resolutions multiple times, such that the patches at different stages cover parts of the image of different real-world-sizes. For a deeper dive into our project, please read the [report](FSDL_Report.pdf).

## Installation
First, clone this repository and (optionally) checkout the branch of our submission by issuing

```bash
git clone git@gitlab.rlp.net:fberres/aerialsegmentation.git
# The following checkout is optional
git checkout -b "final-submission" e1c39e7505f04df8202ae535dc7cd185eaa24f4b
```

Then create a virtual environment, activate it and run the setup bash script to install all dependencies:

```bash
cd aerialsegmentation
mkdir ASVenv
python3 -m venv ASVenv
source ASVenv/bin/activate
pip install --upgrade pip
bash setup.sh
```

Afterwards run the download_data.py script to download and chip the dataset. This may take a while and yields about 21 GB of data after compression. The *\*.tar.gz*-files can be deleted afterwards.

```
python download_data.py
```

Rename the config_template.dat file to config.dat and set the num_worker count ideal for your machine. When using the *--read_config* flag the script will automatically set the num_workers to the count defined in the config.dat file. Alternatively, the number of workers can be set via the *--num_workers* flag instead of using *--read_config*.

## Train
From the project folder run the run_experiment.py script. To replicate and retrain the model we submitted to the wandb benchmark, run the the script with the following arguments.

```
python training/run_experiment.py --wandb --batch_size=8 --data_class=AerialData --dataset=dataset-medium --elevation_alpha=10 --encoder_name=mobilenet_v2 --gpus=1 --loss=cross_entropy --lr=0.0001 --model_class=Unet --augmentations hflip vflip rotate
```

## Inference
To run inference use the run_inference_on_split.py script.

```
python training/run_inference_on_split.py --dataset=dataset-medium --run_id=9815f5bz --metric=val_f1 --project=aerialsegmenation-sweeps --size=4000 --smoothing=1 --stride=7 --split test --batchsize=1
```

With the run_id, project and metric flag one can determine the model used for inference. If the run_id flag is not set the best model in the project according to the given metric is used.


## Submission
For our final submission we used the run_experiment_and_inference.py script.
If you would like to replicate our submission run the script with the following arguments. Note however, that due to the parallelization you will probably not obtain exactly the same results as we did, i.e. the performance might vary slightly.

***Also, be warned***: We used a GeForce GTX 1080 Ti with 11 GB of RAM for training and inference. Not only does this need some time (in our case about 3.5 hours) but the inference is especially memory-intensive, as we chose the maximum size for inference that fit into memory. Hence, if you do not have sufficient RAM on your GPU (or CPU) you should decrease the size used for inference. 
```
training/run_experiment_and_inference.py --wandb --batch_size=8 --data_class=AerialData --dataset=dataset-medium --elevation_alpha=10 --encoder_name=mobilenet_v2 --gpus=1 --loss=cross_entropy --lr=0.0001 --model_class=Unet --augmentations hflip vflip rotate --read_config --inference_size=4000 --smoothing=1 --inference_stride=7 --inference_split test --inference_batchsize=1
```

If you want to learn more about our submission, check out our [submitted run](https://wandb.ai/dronedeploy/dronedeploy-aerial-segmentation/runs/mlvv73do/overview) and our [report](FSDL_Report.pdf).<br/>

## License
[MIT](https://choosealicense.com/licenses/mit/)
