import torch
import torch.nn as nn
from torch.nn import Transformer

"""
Including some experiments network for phase sub-network, haven't gained better precision...
    - Transformer decoder-only part for phase
    - Residual and MLP for phase 
"""

class DecoderOnlyTransformer(torch.nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, device=None):
        super(DecoderOnlyTransformer, self).__init__()
        self.device = device
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=0.0)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = torch.nn.Linear(d_model, 1)
        self.tok_emb = torch.nn.Embedding(5, d_model)
        self.pos_emb = torch.nn.Embedding(64, d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        x[x == -1] = 0
        # x is of shape (batch_size, sequence_length)
        pos = torch.arange(0, x.shape[-1], dtype=torch.int, device=self.device)
        # pos = torch.arange(0, x.shape[-1], dtype=torch.int)

        x = self.tok_emb(x.int()) + self.pos_emb(pos)
        # new x is of shape (batch_size, sequence_length, d_model)
        # but Transformer expects (sequence_length, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # The output of Transformer is of shape (sequence_length, batch_size, d_model)
        # We take the first sequence element (the "start" token) to be the output
        # output = self.transformer.decoder(x, torch.zeros_like(x))
        output = self.decoder(x, torch.zeros_like(x))

        # We pass the output through a final linear layer to get our predicted y
        # The final output is of shape (batch_size, 1)
        return self.fc(output[-1])
        # return self.fc(output).sum(axis=0)

def test_DecoderOnlyTransformer():
    # Create the model
    model = DecoderOnlyTransformer(d_model=512, nhead=8, num_layers=6, dim_feedforward=2048)

    # Create a sample input vector x of shape (batch_size, sequence_length, d_model)
    x = torch.randint(10, (5,)).type(torch.int)
    x = torch.Tensor([[0,0,0,0,0,-1,0], [0,1,2,3,1,0,1]]).type(torch.int)
    print(x)

    # Forward pass through the model
    y = model(x)

    print(y.shape)  # should print: torch.Size([10, 1])
    print(y)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x += identity
        x = self.relu(x)
        return x

class ResidualMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_blocks):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_features, hidden_features) for _ in range(num_blocks)])
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc2(x)
        return x

def test_ResidualMLP():
    # Example usage
    in_features = 100
    hidden_features = 256
    out_features = 10
    num_blocks = 5

    net = ResidualMLP(in_features, hidden_features, out_features, num_blocks)
    print(net)
    input_tensor = torch.randn(32, in_features)
    output = net(input_tensor)
    print(output.shape)

if __name__ == "__main__":
    test_ResidualMLP()
    test_DecoderOnlyTransformer()
