# Define Custom CNN Model for this problem
class DeepCustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepCustomCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Grayscale input with 1 channel
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)  # Assuming input images are 224x224
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Dropout layer to help with regularization
        self.dropout = nn.Dropout(p=0.5)  # Dropout rate of 50%

    def forward(self, x):
        # Forward pass through the network

        # Convolutional and BatchNorm layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 512 * 7 * 7)

        # Fully connected layers with ReLU, dropout, and final output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation since BCEWithLogitsLoss expects raw logits

        return x

# Check for GPU and move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deep_custom_cnn = DeepCustomCNN(num_classes=37).to(device)

def print_model_summary(model, input_size):
    x = torch.randn(1, *input_size).to(device)
    print(f"{'Layer':<30} {'Output Shape':<25} {'# Params':<15}")
    print("=" * 70)

    total_params = 0
    for layer in model.children():
        try:
            x = layer(x)
            num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            total_params += num_params
            print(f"{str(layer):<30} {str(list(x.shape)): <25} {num_params:<15}")
        except Exception as e:
            print(f"{layer}: Layer caused error: {e}")

    print(f"\nTotal trainable parameters: {total_params:,}")

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deep_custom_cnn = DeepCustomCNN(num_classes=37).to(device)

# Print the model summary
print_model_summary(deep_custom_cnn, input_size=(1, 224, 224))  # For grayscale 224x224 image input
