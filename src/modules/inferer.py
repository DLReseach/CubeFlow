import torch

class Inferer:
    def __init__(
        self,
        model,
        optimizer,
        loss,
        dataset,
        saver,
        config
    ):
        super(Inferer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataset = dataset
        self.saver = saver
        self.config = config
        gpu = self.config.gpulab_gpus if self.config.gpulab else self.config.gpus
        self.device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')        
        self.model.to(self.device)

    def infer(self, model_path):
        loss = []
        self.create_dataloader()
        self.dataset.shuffle()
        dl_iter = iter(self.dataloader)
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.model.eval()
        with torch.no_grad():
            for i in range(len(self.dataloader)):
                x, y, comparisons, energy, event_length, file_number = next(dl_iter)
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model.forward(x)
                loss.append(self.loss(y_hat, y))
                self.saver.on_val_step(x, y, y_hat, comparisons, energy, event_length, file_number)
        self.saver.on_val_end()
        avg_loss = torch.stack(loss).mean()
        return avg_loss

    def create_dataloader(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1024,
            num_workers=4,
            drop_last=True,
            shuffle=False
        )