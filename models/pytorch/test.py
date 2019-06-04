import torch
from warmup_scheduler import GradualWarmupScheduler
from torch.optim import lr_scheduler

v = torch.zeros(10)
optim = torch.optim.SGD([v], lr=0.01)

#scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, verbose=True)
#scheduler = GradualWarmupScheduler(optim, multiplier=8, total_epoch=10, after_scheduler=scheduler_plateau)

#scheduler = lr_scheduler.CosineAnnealingLR(optim,
#                                           20, eta_min=1e-4)
scheduler = lr_scheduler.StepLR(optim, 20, decay=0.1)

for epoch in range(1, 100):
    scheduler.step()
    print(epoch, optim.param_groups[0]['lr'])
