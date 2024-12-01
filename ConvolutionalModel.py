import torch
import torch.nn as nn
import torch.nn.functional as F

multiply = lambda x: x[0]*multiply(x[1:]) if len(x) > 1 else x[0]

class ConvolutionalModel(nn.Module):
    def __init__(self, stride, kernel_size, batch_size, RecursiveModel):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        self.recursive_forward_function = RecursiveModel
        input_size = RecursiveModel.input_size
        output_size = RecursiveModel.output_size
        RecursiveModel.Block_layer_modification = True
        RecursiveModel.set_ConvolutionalModel(self)

        self.expanding = nn.Linear(self.kernel_size ** 2, multiply(input_size)) #or nn.Upsample(size=input_size, mode='bilinear',align_corners=False)
        self.attention = nn.Linear(self.kernel_size ** 2, 1) #or sequential(......)
        self.resize = nn.Linear(multiply(output_size), self.kernel_size ** 2)

        self.batch_size = batch_size
        self.batches = {} # format: {self.encode(....): chunk}

        self.addresses = []
        self.addresses_counter = -1
    def encode(self, *items): # batch_idx, i, j,floor,address_idx
        return '_'.join([str(item) for item in items])

    def decode(self, encoded):

        return tuple(map(int,encoded.split("_")))

    def process_batch(self):
        for i in range(self.batch_size):
            key,window =  self.batches.popitem()
            batch_idx, i, j,floor,address_idx = self.decode(key)

            attention = F.relu(self.attention(window))
            x = self.addresses[address_idx]

            expanded_window = self.expanding(window)
            expanded_window = expanded_window.view(-1, *input_size)

            recursive_output = self.recursive_forward_function(expanded_window, floor + 1)

            recursive_output = recursive_output.view(-1, multiply(output_size))
            recursive_output = self.resize(recursive_output)
            replacement = recursive_output.view(-1, self.kernel_size, self.kernel_size)

            x[batch_idx, :, i:i + self.kernel_size, j:j + self.kernel_size] += replacement * attention

    def add_to_batch(self, batch, i, j, floor,chunk, addresses_counter):
        self.batches[self.encode(batch, i, j,floor, addresses_counter)] = chunk
        if len(self.batches) >= self.batch_size:
            self.process_batch()
            self.batches = {}
    def set_recursive_forward(self, model):
        self.recursive_forward_function = model

    def forward(self, x,current_recursion_floor):
        if self.recursive_forward_function is None:
            raise NotImplementedError
        batch_size, channels, height, width = x.size()
        self.addresses_counter+=1
        self.addresses.append(x)
        for batch_idx in range(batch_size):
            for i in range(0, height - self.kernel_size + 1, self.stride):
                for j in range(0, width - self.kernel_size + 1, self.stride):
                    window = x[batch_idx, :, i:i + self.kernel_size, j:j + self.kernel_size]
                    window = window.reshape(-1, self.kernel_size ** 2)
                    threshold = F.relu(self.attention(window))
                    if threshold > 0:
                        self.add_to_batch(batch_idx, i, j,current_recursion_floor, window, self.addresses_counter)


        return x
