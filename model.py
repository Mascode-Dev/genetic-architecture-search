# model.py
import torch
import torch.nn as nn
import math

class DynamicTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, config_genes):
        super().__init__()
        
        # Extraction des gènes
        self.d_model = config_genes['d_model']
        self.n_heads = config_genes['n_heads']
        self.n_layers = config_genes['n_layers']
        ratio = config_genes['dim_feedforward_ratio']
        self.dim_feedforward = self.d_model * ratio

        # Embedding (Projection simple pour l'exemple)
        # Pour du texte réel, ce serait nn.Embedding
        self.embedding = nn.Linear(input_dim, self.d_model)
        
        # Le coeur du Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Tête de classification finale
        self.classifier = nn.Linear(self.d_model, output_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        x = self.embedding(x)
        
        # Positional Encoding simplifié (optionnel pour ce test rapide)
        
        # Passage dans le Transformer
        x = self.transformer(x)
        
        # On prend la moyenne sur la séquence (Global Average Pooling)
        x = x.mean(dim=1)
        
        return self.classifier(x)

    def count_parameters(self):
        """Compte les paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)