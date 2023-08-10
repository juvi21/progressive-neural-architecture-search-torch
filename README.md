# progressive-neural-architecture-search-torch
Unofficial PyTorch implementation of Progressive Neural Architecture Search. https://arxiv.org/pdf/1712.00559.pdf

<img src="assets/architecturePNAS.png" alt="architecturePNAS" width="600"/>


### To do:
- [ ] Add Unittests
- [ ] Push PyPi
- [ ] Better and easier data cleaning
- [ ] Better README

## USAGE

See: [train.py](train.py)

```python
import torch
from src.pnasnet import PNASNet

model = PNASNET(num_classes=5).to(device)

train_img_folder = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_img_folder, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)

lr = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_fn = nn.NLLLoss().to(device)

model.train()
for epoch in range(2):
    train_one_epoch(model, optimizer, train_loader, loss_fn)
```
