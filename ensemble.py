# Define the EnsembleModel
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, weight1=0.5, weight2=0.5):
        """
        Initialize the ensemble model with two models and their respective weights.

        :param model1: First pre-initialized model (e.g., ResNet50)
        :param model2: Second pre-initialized model (e.g., DeepCustomCNN)
        :param weight1: Weight for the output of model1 (default: 0.5)
        :param weight2: Weight for the output of model2 (default: 0.5)
        """
        super(EnsembleModel, self).__init__()
        self.model1 = model1  # Pre-initialised model 1 (e.g., ResNet50)
        self.model2 = model2  # Pre-initialised model 2 (e.g., DeepCustomCNN)
        self.weight1 = weight1  # Weight for model1
        self.weight2 = weight2  # Weight for model2

    def forward(self, x):
        # Move input to the device of model1 (assuming both models are on the same device)
        x = x.to(next(self.model1.parameters()).device)

        # Forward pass through both models
        out1 = self.model1(x)
        out2 = self.model2(x)

        # Return the weighted average of the outputs
        ensemble_output = (self.weight1 * out1) + (self.weight2 * out2)
        return ensemble_output


# Move both models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet50 = resnet50.to(device)
deep_custom_cnn = deep_custom_cnn.to(device)

# Initialise the ensemble model with weighted averaging (optional)
ensemble_model = EnsembleModel(resnet50, deep_custom_cnn, weight1=0.6, weight2=0.4).to(device)
