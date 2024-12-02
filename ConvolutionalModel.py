import torch
import torch.nn as nn
import torch.nn.functional as F
from Debug import debug

multiply = lambda x: x[0]*multiply(x[1:]) if len(x) > 1 else x[0]
THRESHOLD = 0


class ConvolutionalModel(nn.Module):
    def __init__(self, stride, kernel_size, batch_size, RecursiveModel):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        self.recursive_forward_function = RecursiveModel
        self.input_size = RecursiveModel.input_size
        self.output_size = RecursiveModel.output_size
        RecursiveModel.Block_layer_modification = True
        RecursiveModel.set_ConvolutionalModel(self)

        self.expanding = nn.Linear(self.kernel_size ** 2, multiply(self.input_size)) #or nn.Upsample(size=self.input_size, mode='bilinear',align_corners=False)
        self.attention = nn.Linear(self.kernel_size ** 2, 1) #or sequential(......)
        self.resize = nn.Linear(multiply(self.output_size), self.kernel_size ** 2)

        self.batch_size = batch_size
        self.batches = [] # format: [self.encode(....)]

        self.addresses = []
        self.addresses_counter = -1
    def encode(self, *items): # batch_idx, i, j,address_idx
        return '_'.join([str(item) for item in items])

    def decode(self, encoded):

        return tuple(map(int,encoded.split("_")))

    def process_batch(self,floor):
        debug([debug.show_reformed_batch_size],f"Processing batch of size {len(self.batches)}")
        codes = []
        reformed_batch = []
        for i in range(self.batch_size):
            if len(self.batches) == 0:
                break
            code =  self.batches.pop()
            codes.append(code)
            batch_idx, i, j,address_idx = self.decode(code)
            x = self.addresses[address_idx]
            window = x[batch_idx, :, i:i + self.kernel_size, j:j + self.kernel_size]
            window = window.reshape(-1, self.kernel_size ** 2)
            reformed_batch.append(window)

        reformed_batch = torch.stack(reformed_batch, dim=0)

        attention = F.relu(self.attention(reformed_batch))

        expanded_window = self.expanding(reformed_batch)
        expanded_window = expanded_window.view(-1, *self.input_size)

        recursive_output = self.recursive_forward_function(expanded_window, floor + 1)

        recursive_output = recursive_output.view(-1, multiply(self.output_size))
        recursive_output = self.resize(recursive_output)
        replacement = recursive_output.view(-1, self.kernel_size, self.kernel_size)

        for code,a,re in zip(codes,attention,replacement):
            batch_idx, i, j,address_idx = self.decode(code)
            x = self.addresses[address_idx]
            x[batch_idx, :, i:i + self.kernel_size, j:j + self.kernel_size] += re * a

    def add_to_batch(self, batch, i, j, addresses_counter,floor):
        self.batches.append(self.encode(batch, i, j,addresses_counter))
        if len(self.batches) >= self.batch_size:
            self.process_batch(floor)
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
                    if threshold > THRESHOLD:
                        self.add_to_batch(batch_idx, i, j, self.addresses_counter,current_recursion_floor)
        if len(self.batches) > 0:
            self.process_batch(current_recursion_floor)

        return x
