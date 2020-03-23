import torch
import pandas as pd

class Inferer:
    def __init__(
        self,
        model,
        optimizer,
        loss,
        dataset,
        config
    ):
        super(Inferer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataset = dataset
        self.config = config
        gpu = self.config.gpulab_gpus if self.config.gpulab else self.config.gpus
        self.device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')        
        self.model.to(self.device)

        torch.multiprocessing.set_sharing_strategy('file_system')

        self.data = {'event': []}
        self.predictions = {target.replace('true', 'own'): [] for target in self.config.targets}
        self.truths = {target: [] for target in self.config.targets}

    def infer(self, model_path, save_path):
        loss = []
        self.create_dataloader()
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                print(batch)
                x = batch[0].to(self.device).float()
                y = batch[1].to(self.device).float()
                y_hat = self.model.forward(x)
                loss.append(self.loss(y_hat, y))
                self.on_test_step(y, y_hat, batch[2])
                # if i == 0:
                #     break
        self.data.update(self.predictions)
        self.data.update(self.truths)
        predictions = pd.DataFrame().from_dict(self.data)
        file_name = save_path.joinpath('predictions.gzip')
        predictions.to_parquet(file_name, engine='fastparquet')

    def on_test_step(self, y, y_hat, events):
        self.data['event'].extend(events)
        for i, target in enumerate(self.config.targets):
            self.predictions[target.replace('true', 'own')].extend(y_hat[:, i])
            self.truths[target].extend(y[:, i])


    def create_dataloader(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=4,
            shuffle=False
        )