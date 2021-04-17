from typing import Any, Dict, List
import argparse
import segmentation_models_pytorch as smp

DECODER_CHANNELS = (256, 128, 64, 32, 16)
DECODER_USE_BATCHNORM = True
ENCODER_NAME = "resnet18"
ENCODER_WEIGHTS = "imagenet"
ENCODER_DEPTH = 5

class Unet(smp.Unet):
    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        
        self.args = vars(args) if args is not None else {}
        self.in_channels = data_config["input_dims"][0]
        self.num_classes = data_config["mapping"][0]
        self.class_labels = data_config["class_labels"]

        decoder_channels = self.args.get("decoder_channels", DECODER_CHANNELS)
        decoder_use_batchnorm = self.args.get("decoder_use_batchnorm", DECODER_USE_BATCHNORM)
        encoder_name = self.args.get("encoder_name", ENCODER_NAME)
        encoder_weights = self.args.get("encoder_weights", ENCODER_WEIGHTS)
        encoder_depth = self.args.get("encoder_depth", ENCODER_DEPTH) 

        super().__init__(in_channels=self.in_channels, classes=self.num_classes, decoder_channels=decoder_channels,
                        decoder_use_batchnorm=decoder_use_batchnorm, encoder_name=encoder_name, encoder_weights=encoder_weights,
                        encoder_depth=encoder_depth)


    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--decoder_channels", type=List[int], default=DECODER_CHANNELS)
        parser.add_argument("--decoder_use_batchnorm", type=bool, default=DECODER_USE_BATCHNORM)
        parser.add_argument("--encoder_name", type=str, default=ENCODER_NAME)
        parser.add_argument("--encoder_weights", type=str, default=ENCODER_WEIGHTS)
        parser.add_argument("--encoder_depth", type=int, default=ENCODER_DEPTH)
        return parser