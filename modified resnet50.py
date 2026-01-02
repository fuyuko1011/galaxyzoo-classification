# Load pre-trained ResNet50 model and move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available

# Load the ResNet50 model with pre-trained weights
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Modify the first convolution layer to accept 1-channel input instead of 3-channel
resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify the final fully connected layer to output 37 classes (or the number of classes in your problem)
resnet50.fc = nn.Linear(resnet50.fc.in_features, 37)

# Move the model to the GPU
resnet50 = resnet50.to(device)

# Include AMP (Automatic Mixed Precision) usage
scaler = torch.amp.GradScaler('cuda')
