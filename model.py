import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.2):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                n = param.size(0)
                chunk = n // 4
                param.data[0:chunk].fill_(-3.0)  # input gate
                param.data[chunk:2*chunk].fill_(1.0)  # forget gate
                param.data[2*chunk:3*chunk].fill_(0.0)  # cell gate
                param.data[3*chunk:4*chunk].fill_(-2.0)  # output gate
        nn.init.xavier_uniform_(self.hidden_to_latent.weight)
        nn.init.zeros_(self.hidden_to_latent.bias)
    
    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        latent = self.hidden_to_latent(h_n[-1])
        return latent, (h_n, c_n)


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim=7, hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.2):
        super(LSTMDecoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                n = param.size(0)
                chunk = n // 4
                param.data[0:chunk].fill_(-3.0)
                param.data[chunk:2*chunk].fill_(1.0)
                param.data[2*chunk:3*chunk].fill_(0.0)
                param.data[3*chunk:4*chunk].fill_(-2.0)
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.zeros_(self.latent_to_hidden.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x_reversed, hidden_state):
        output, _ = self.lstm(x_reversed, hidden_state)
        return self.output_layer(output)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
    
    def forward(self, x):
        latent, hidden_state = self.encoder(x)
        x_reversed = torch.flip(x, dims=[1])
        reconstruction_reversed = self.decoder(x_reversed, hidden_state)
        reconstruction = torch.flip(reconstruction_reversed, dims=[1])
        return reconstruction, latent


if __name__ == "__main__":
    torch.manual_seed(42)
    dummy_input = torch.randn(32, 60, 7)
    model = LSTMAutoencoder()
    
    model.eval()
    with torch.no_grad():
        reconstruction, latent = model(dummy_input)
    
    print(f"Input: {dummy_input.shape}")
    print(f"Reconstruction: {reconstruction.shape}")
    print(f"Latent: {latent.shape}")
    print(f"MSE Loss: {nn.MSELoss()(reconstruction, dummy_input).item():.4f}")
