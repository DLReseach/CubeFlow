import torch


class Adam:
	def __init__(self, model_parameters, min_learning_rate):
		return torch.optim.Adam(model_parameters, lr=min_learning_rate)
