import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from src.pnasnet import PNASNet

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/data_new')


def mytransform(img: Image.Image) -> Image.Image:
    width, height = img.size
    pad_to = max(width, height)
    transformed = torchvision.transforms.functional.pad(img, (pad_to - width, pad_to - height))
    transformed = torchvision.transforms.functional.resize(transformed, (331, 331))
    return transformed


def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, loss_fn: nn.Module) -> (float, int):
    loss = 0
    num_correct = 0
    with torch.no_grad():
        model.eval()
        for idx, (input, target) in enumerate(loader):
            out = model(input.to(device))
            target = target.to(device)
            loss += loss_fn(m(out), target).item()
            predictions = torch.argmax(out, dim=1)
            num_correct += torch.sum(target == predictions).item()
        model.train()
    return loss, num_correct


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loader: torch.utils.data.DataLoader, loss_fn: nn.Module) -> None:
    log_softmax = nn.LogSoftmax(dim=1).to(device)  # Add this line
    for idx, (input, target) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        out = model(input.to(device))
        loss = loss_fn(log_softmax(out), target.to(device))  # Modify this line

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    setup_seed()

    transform = transforms.Compose([
        mytransform,
        transforms.ToTensor()
    ])

    model = PNASNet(num_classes=5).to(device)
    model = torch.jit.script(model)

    train_img_folder = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_img_folder, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)

    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    m = nn.LogSoftmax(dim=1).to(device)
    loss_fn = nn.NLLLoss().to(device)

    model.train()
    for epoch in range(2):
        train_one_epoch(model, optimizer, train_loader, loss_fn)

    sample_input, _ = next(iter(train_loader))
    sample_out = model(sample_input.to(device))

    fixture = torch.tensor([
        [-0.6241,  1.2502,  1.1059, -0.6813, 0.8837],
        [1.3981,  0.7711, -0.3744,  0.6056, -0.7757],
        [0.2892, -0.0913, -0.3163,  0.9425, -0.7759],
        [-0.1292,  1.3259, -0.1468,  0.0087,  0.3588]
    ], device=device)


if __name__ == "__main__":
    main()