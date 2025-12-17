"""
LSTM/GRU Baseline Model (Part 2b)
A recurrent neural network baseline for error recognition.
"""
import torch
import torch.nn as nn
from core.models.blocks import fetch_input_dim


class LSTMErrorRecognition(nn.Module):
    """
    LSTM-based model for error recognition.
    Processes the sequence of sub-step features and predicts error probability.
    """
    
    def __init__(self, config, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=True):
        super(LSTMErrorRecognition, self).__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        input_dim = fetch_input_dim(config)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output dimension based on bidirectional setting
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
        
        Returns:
            Output tensor of shape (batch_size, 1) - error probability logits
        """
        # Handle input shape - add sequence dimension if needed
        if x.dim() == 2:
            # Input is (batch_size, feature_dim) - treat each sample as single timestep
            x = x.unsqueeze(1)  # (batch_size, 1, feature_dim)
        
        # Handle NaN values
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state from both directions if bidirectional
        if self.bidirectional:
            # Concatenate the last hidden states from both directions
            hidden_forward = hidden[-2, :, :]  # Last layer, forward
            hidden_backward = hidden[-1, :, :]  # Last layer, backward
            hidden_combined = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            hidden_combined = hidden[-1, :, :]  # Last layer hidden state
        
        # Classification
        output = self.classifier(hidden_combined)
        
        return output


class GRUErrorRecognition(nn.Module):
    """
    GRU-based model for error recognition.
    Similar to LSTM but uses GRU cells which are computationally lighter.
    """
    
    def __init__(self, config, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=True):
        super(GRUErrorRecognition, self).__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        input_dim = fetch_input_dim(config)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output dimension based on bidirectional setting
        gru_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
        
        Returns:
            Output tensor of shape (batch_size, 1) - error probability logits
        """
        # Handle input shape - add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Handle NaN values
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Use the last hidden state from both directions if bidirectional
        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_combined = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            hidden_combined = hidden[-1, :, :]
        
        # Classification
        output = self.classifier(hidden_combined)
        
        return output


class LSTMSequenceErrorRecognition(nn.Module):
    """
    LSTM model that outputs predictions for each timestep in the sequence.
    This allows for sub-step level predictions like the MLP baseline.
    """
    
    def __init__(self, config, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=True):
        super(LSTMSequenceErrorRecognition, self).__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        input_dim = fetch_input_dim(config)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output dimension
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Per-timestep classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        """
        Forward pass - outputs prediction for each timestep.
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim)
               Each sample is a single sub-step feature
        
        Returns:
            Output tensor of shape (batch_size, 1) - per sub-step error probability logits
        """
        # Handle NaN values
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # For compatibility with existing training loop that passes individual sub-steps
        # We treat each sample independently
        if x.dim() == 2:
            # Add sequence dimension
            x = x.unsqueeze(1)  # (batch_size, 1, feature_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden*2)
        
        # Apply classifier to each timestep output
        output = self.classifier(lstm_out)  # (batch_size, seq_len, 1)
        
        # Squeeze sequence dimension if it's 1
        if output.size(1) == 1:
            output = output.squeeze(1)  # (batch_size, 1)
        
        return output

