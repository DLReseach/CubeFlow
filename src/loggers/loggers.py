from pytorch_lightning.logging import LightningLoggerBase

class WandbLogger():
    def __init__(self, wandb):
        super().__init__()
        self.wandb = wandb
        self.reset()

    def reset(self):
        self.metrics = []


    def update(self, value):
        self.metrics.append(value)


    def log_metrics(self):
        print(self.metrics)
        for metric in self.metrics:
            avg_metric = torch.stack([x[metric] for x in self.metrics]).mean()
            self.wandb.log({metric: avg_metric})
            print('logged')
