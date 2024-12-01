from RecursiveModel import Recursive_model,multiply
from ConvolutionalModel import ConvolutionalModel
import torch

batch_size = 10
input_size = (1, 64, 64)


model = Recursive_model(input_size, 2)

model.add_convolutional_layer(1, 3,5, 3)
model.add_convolutional_layer(3, 3,3, 1)
model.add_linear_layer(16)
model.add_resize_layer((1,4,4))

ConvolutionalModel(stride=4, kernel_size=4, batch_size=50,RecursiveModel=model)

random_testing_input_matrix = torch.randn(batch_size, *input_size)
output = model(random_testing_input_matrix)
print(output.shape)
