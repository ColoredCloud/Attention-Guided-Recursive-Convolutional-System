# Description
#### The system recursively uses models for convolution operations on inputs, coupled with simple attention mechanisms to deal with priorities and points that need to be subdivided. 
#### The model is designed to deal with and solve both logical and mathematical problems,as it is common to think about mathematical problems by converting them into several simple problems or combinations of definitions.

# Usage

To use the models, you can follow these steps:

## 1. Import the necessary modules:

```python
import torch
from RecursiveModel import RecursiveModel
from ConvolutionalModel import ConvolutionalModel
```

## 2. Create instances of the RecursiveModel:

```python
batch_size = 10
input_size = (1, 64, 64)

# Create a RecursiveModel instance
model = RecursiveModel(input_size, 2)

# Add layers to the model
model.add_convolutional_layer(1, 3, 5, 3)
model.add_convolutional_layer(3, 3, 3, 1)
model.add_linear_layer(16)
model.add_resize_layer((1, 4, 4))
```
## 3. Create a ConvolutionalModel instance and link it to the RecursiveModel:
```python
# Create a ConvolutionalModel instance
ConvolutionalModel(stride=1, kernel_size=3, batch_size=100,RecursiveModel=model)
```

## 4. Enable debug and test the model by using a random matrix:

```python
from Debug import debug
debug.show_reformed_batch_size = True
random_testing_input_matrix = torch.randn(batch_size, *input_size)
output = model(random_testing_input_matrix)
print(output.shape)
```

## Other features
In the ConvolutionalModel, the model uses an attention mechanism to deal with priorities and points that need to be subdivided. If a chunk's score passes a threshold, it will be stored in a list, wait to be assymbled with other chunks and process. This mechanism is designed to facilitate efficient and effective use of GPU resources.

The RecursiveModel can also be used with other models or Layers. 

Also, it is possible to use different Recursive Model in different floor of the Recursion process.
You can add these to the model by using a subclass of RecursiveModel and add functions like add_rnn_layer or add_gru_layer.


# Contribution

If you would like to contribute to this project, please follow these steps:

1. Fork the repository and clone it to your local machine.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them to your branch.
4. Push your branch to your fork of the repository.
5. Submit a pull request to the original repository.

Please ensure that your contributions follow the existing code style and include appropriate unit tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.