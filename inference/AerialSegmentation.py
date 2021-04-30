from models.unet import Unet


class SegModel:
  """
  Class to run inference at the end of testing
  """

  def __init__(self, lit_model, data_config, args):
    data_config = data_config 
    model = Unet(data_config=data_config, args=args)
    self.lit_model = BaseLitModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, args=args, model=model
        )
    self.lit_model.eval()
    self.scripted_model = self.lit_model.to_torchscript(method="script", file_path=None)

    @torch.no_grad()
    def predict(self, image):
        """
        Prediction function for benchmark prediciton at the end of the training process
        """
        logits = self.scripted_model(x)
        probs = F.softmax(logits, dim=1)
        pred = torch.topk(y_pred_softmax, 1)[1]
        
        return pred
