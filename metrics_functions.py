import torch

# Подключим видеокарту!
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)


def compute_accuracy(model, loader):
    """
    Computes accuracy on the dataset wrapped in a loader

    Returns: accuracy as a float value between 0 and 1
    """
    model.eval()  # Evaluation mode

    total_samples = 0
    true_samples = 0

    for i_step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        prediction = torch.argmax(model(x), dim=1)
        true_samples += int(loader.batch_size - torch.count_nonzero(prediction - y))
        total_samples += loader.batch_size

    return float(true_samples / total_samples)
