from typing import Any, Dict, List
import argparse
import segmentation_models_pytorch as smp
import math
from torch.nn import ZeroPad2d
from torchvision.transforms import CenterCrop

DECODER_CHANNELS = (256, 128, 64, 32, 16)
DECODER_USE_BATCHNORM = True
ENCODER_NAME = "resnet18"
ENCODER_WEIGHTS = "imagenet"
ENCODER_DEPTH = 5
FREEZE_ENCODER = False

class Unet(smp.Unet):
    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:

        self.args = vars(args) if args is not None else {}
        self.in_channels = data_config["input_dims"][0]
        self.image_size = data_config["input_dims"][1]
        self.num_classes = data_config["mapping"][0]
        self.class_labels = data_config["class_labels"]

        # If elevation is to be predicted, too, add another 'class',
        # which will serve as elevation prediction
        self.predict_elevation = (self.args.get("elevation_alpha", 0.0) > 0.0)
        if self.predict_elevation:
            self.num_classes += 1


        decoder_channels = self.args.get("decoder_channels", DECODER_CHANNELS)
        decoder_use_batchnorm = self.args.get("decoder_use_batchnorm", DECODER_USE_BATCHNORM)
        encoder_name = self.args.get("encoder_name", ENCODER_NAME)
        encoder_weights = self.args.get("encoder_weights", ENCODER_WEIGHTS)
        encoder_depth = self.args.get("encoder_depth", ENCODER_DEPTH)
        freeze_encoder = self.args.get("freeze_encoder", FREEZE_ENCODER)

        self.padding = (math.ceil(self.image_size / 2**(encoder_depth))*2**(encoder_depth)-self.image_size)//2

        super().__init__(in_channels=self.in_channels, classes=self.num_classes, decoder_channels=decoder_channels,
                        decoder_use_batchnorm=decoder_use_batchnorm, encoder_name=encoder_name, encoder_weights=encoder_weights,
                        encoder_depth=encoder_depth)
        if freeze_encoder:
            self.freeze_encoder()


    def freeze_encoder(self):
        for name, param in self.named_parameters():
            if "encoder" in name:
                param.requires_grad = False

    def pad(self, x):
        return ZeroPad2d(self.padding)(x)

    def forward(self, x):
        x = self.pad(x)
        features = super().forward(x)
        features = CenterCrop(self.image_size)(features)
        return features


    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--decoder_channels", type=List[int], default=DECODER_CHANNELS)
        parser.add_argument("--decoder_use_batchnorm", type=bool, default=DECODER_USE_BATCHNORM)
        parser.add_argument("--encoder_name", type=str, default=ENCODER_NAME)
        parser.add_argument("--encoder_weights", type=str, default=ENCODER_WEIGHTS)
        parser.add_argument("--encoder_depth", type=int, default=ENCODER_DEPTH)
        parser.add_argument("--freeze_encoder", action="store_true", default=False)
        return parser
