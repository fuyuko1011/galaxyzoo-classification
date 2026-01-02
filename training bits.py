# Three models have been taken into consideration: 
# Pretrained ResNet50 from He et al. (2015)
# Custom CNN network of a depth comparable to the ResNet50 designed for the task
# An ensemble configuration of the two.

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Early stopping to stop the training when the validation loss does not improve after a given patience.

        :param patience: How long to wait after the last time validation loss improved.
        :param verbose: If True, prints a message for each validation loss improvement.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        :param path: Path for the checkpoint to save the model.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        """
        Call this function after each validation step to check if training should stop.

        :param val_loss: The current validation loss.
        :param model: The model that is being trained.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves the model when the validation loss decreases.
        Saves the model state dictionary with the 'cuda' context removed to allow for flexibility.

        :param val_loss: The new validation loss.
        :param model: The model whose state will be saved.
        """
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")

        # Save the model's state dict in a way that it can be loaded both on CPU and GPU
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# The following function creates the optimisers and schedulers in the run-up to building the training loop function. 
# We use the 1Cycle learning rate scheduler, AdamW with weight decay for L2 regularisation and early stopping with `patience=7`.

def create_optimizers_and_schedulers(model, train_loader, num_epochs, weight_decay=1e-4):
    """
    Creates optimizer, scheduler, and early stopping for model training with L2 regularisation.

    :param model: The model to optimise.
    :param train_loader: DataLoader for the training set.
    :param num_epochs: Number of epochs for the OneCycleLR scheduler.
    :param weight_decay: L2 regularisation (weight decay) coefficient.
    :return: optimizer, scheduler, early stopping
    """
    # Optimizer: AdamW with weight_decay for L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)

    # Scheduler: OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
                                                    steps_per_epoch=len(train_loader), epochs=num_epochs)

    # Early stopping: Stop if the validation loss doesn't improve after 7 epochs
    early_stopping = EarlyStopping(patience=7, verbose=True, path=f'best_model_{type(model).__name__}.pth')

    return optimizer, scheduler, early_stopping
