import torch


class Mse:
	def __init__(self):
		self.loss = torch.nn.MSELoss()
