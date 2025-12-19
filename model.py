# model.py
import torch
import torch.nn as nn
import math

class DynamicTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, config_genes):
        super().__init__()
        
        # Genes extraction
        self.d_model = config_genes['d_model'] # Model dimension
        self.n_heads = config_genes['n_heads'] # Number of attention heads
        self.n_layers = config_genes['n_layers'] # Number of Transformer layers
        ratio = config_genes['dim_feedforward_ratio'] # Ratio for feedforward dimension
        self.dim_feedforward = self.d_model * ratio

        # Embedding (Simple projection for the example)
        # For real text, this would be nn.Embedding
        self.embedding = nn.Linear(input_dim, self.d_model)
        
        # The heart of the Transformer - Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Final classification head
        self.classifier = nn.Linear(self.d_model, output_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        x = self.embedding(x)
        
        # Transformer Encoder
        x = self.transformer(x)
        
        # We take the mean over the sequence (Global Average Pooling)
        x = x.mean(dim=1)
        
        return self.classifier(x)

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)