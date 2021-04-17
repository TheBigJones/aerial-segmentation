import segmentation_models_pytorch as smp 

class Unet(smp.Unet):
    #TODO Build a Unet class that inherits from smp.Unet in order to use argparsing for setting the amount of decoder channels or encoder depth 