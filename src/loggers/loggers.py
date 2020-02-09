import torch


class WandbHelper():
    def __init__(self, logger, metric_types):
        super().__init__()
        self.logger = logger
        self.metric_types = metric_types
        self.reset()


    def reset(self):
        self.metrics = {key: [] for key in self.metric_types}


    def update(self, value_dict):
        metric = list(value_dict.keys())[0]
        value = list(value_dict.values())[0]
        self.metrics[metric].append(value)


    def log_metrics(self, step):
        for metric in self.metric_types:
            avg_metric = torch.stack(self.metrics[metric]).mean()
            self.logger.log_metrics({metric: avg_metric}, step=step)
