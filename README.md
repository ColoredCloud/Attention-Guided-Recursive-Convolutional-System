## Usage

To use the models, you can follow these steps:

1. Import the necessary modules:

```python
import torch
from RecursiveModel import RecursiveModel
from ConvolutionalModel import ConvolutionalModel
```

2. Create instances of the models:

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

# Create a ConvolutionalModel instance
convolutional_model = ConvolutionalModel(stride=4, kernel_size=4, batch_size=50, RecursiveModel=model)
```

3. Generate random input data:

```python
random_testing_input_matrix = torch.randn(batch_size, *input_size)
```

4. Run the model:

```python
output = model(random_testing_input_matrix)
print(output.shape)
```

## Contribution

If you would like to contribute to this project, please follow these steps:

1. Fork the repository and clone it to your local machine.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them to your branch.
4. Push your branch to your fork of the repository.
5. Submit a pull request to the original repository.

Please ensure that your contributions follow the existing code style and include appropriate unit tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.