from RecursiveModel import Recursive_model,multiply
from ConvolutionalModel import ConvolutionalModel
import torch


batch_size = 10
input_size = (1, 64, 64)


model = Recursive_model(input_size, 3)

model.add_convolutional_layer(1, 3,5, 3)
model.add_convolutional_layer(3, 3,3, 1)
model.add_linear_layer(16)
model.add_resize_layer((1,4,4))

ConvolutionalModel(stride=1, kernel_size=3, batch_size=100,RecursiveModel=model)

from Debug import debug
debug.show_reformed_batch_size = True
random_testing_input_matrix = torch.randn(batch_size, *input_size)
output = model(random_testing_input_matrix)
print(output.shape)
