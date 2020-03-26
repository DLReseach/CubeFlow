import torch


class Mse:
	def __init__(self):
		return torch.nn.MSELoss()
