import torch


class Adam:
	def __init__(self, model_parameters, min_learning_rate):
		self.model_parameters = model_parameters
		self.min_learning_rate = min_learning_rate
		self.optimizer = torch.optim.Adam(self.model_parameters, lr=self.min_learning_rate)
