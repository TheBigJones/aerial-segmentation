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


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)