import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
import config_meld as config
from trhgt import TRHGTConv


class HGTEmotionRecognizer(nn.Module):
    def __init__(self, feature_dims, hidden_channels, out_channels,
                 hgt_num_heads, hgt_num_layers, dropout_rate,
                 num_speakers, speaker_emb_dim,
                 transformer_nhead, transformer_num_layers,
                 transformer_ff_multiplier, transformer_activation,
                 classifier_hidden_dim_multiplier):
        super().__init__()

        self.node_types = config.NODE_TYPES
        self.edge_types = config.EDGE_TYPES
        self.metadata = (self.node_types, self.edge_types)
        self.hidden_channels = hidden_channels
        self.num_hgt_layers = hgt_num_layers

        if hidden_channels % hgt_num_heads != 0:
            raise ValueError(f"HGT hidden_channels ({hidden_channels}) must be divisible by hgt_num_heads ({hgt_num_heads}).")
        if hidden_channels % transformer_nhead != 0:
            raise ValueError(f"Transformer hidden_channels ({hidden_channels}) must be divisible by transformer_nhead ({transformer_nhead}).")

        self.speaker_embedding = nn.Embedding(num_speakers, speaker_emb_dim)

        self.input_proj_dict = nn.ModuleDict()
        for node_type in self.node_types:
            self.input_proj_dict[node_type] = Linear(feature_dims[node_type] + speaker_emb_dim, hidden_channels)

        self.transformer_encoders = nn.ModuleDict()
        for node_type in self.node_types:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channels, nhead=transformer_nhead,
                dim_feedforward=hidden_channels * transformer_ff_multiplier,
                dropout=dropout_rate, activation=transformer_activation, batch_first=True
            )
            self.transformer_encoders[node_type] = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_hgt_layers):
            conv = TRHGTConv(hidden_channels, hidden_channels, self.metadata, hgt_num_heads)
            self.convs.append(conv)
            norm_dict = nn.ModuleDict()
            for node_type in self.node_types:
                norm_dict[node_type] = nn.LayerNorm(hidden_channels)
            self.norms.append(norm_dict)

        self.dropout = nn.Dropout(dropout_rate)

        num_modalities = len(self.node_types)
        classifier_input_dim = hidden_channels * num_modalities
        classifier_hidden_dim = int(classifier_input_dim * classifier_hidden_dim_multiplier)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, out_channels),
        )

    def forward(self, x_dict, edge_index_dict, speaker_idx_tensor, batch_dict):
        enriched_x_dict = {}
        speaker_embs = self.speaker_embedding(speaker_idx_tensor)

        for node_type, x_features in x_dict.items():
            x_with_speakers = torch.cat([x_features, speaker_embs], dim=-1)
            projected_features = self.input_proj_dict[node_type](x_with_speakers)
            
            batch = batch_dict[node_type]
            dialogue_outputs = []
            for dialogue_idx in torch.unique(batch):
                dialogue_mask = (batch == dialogue_idx)
                dialogue_sequence = projected_features[dialogue_mask].unsqueeze(0)
                transformer_out = self.transformer_encoders[node_type](dialogue_sequence)
                dialogue_outputs.append(transformer_out.squeeze(0))
            
            enriched_x_dict[node_type] = torch.cat(dialogue_outputs, dim=0)

        current_x_dict = enriched_x_dict
        for conv, norm_dict in zip(self.convs, self.norms):
            current_x_dict = conv(current_x_dict, edge_index_dict, batch_dict=batch_dict)
            current_x_dict = {
                nt: self.dropout(norm(x).relu())
                for nt, x, norm in zip(current_x_dict.keys(), current_x_dict.values(), norm_dict.values())
            }
        
        list_of_final_embeddings = [current_x_dict[nt] for nt in self.node_types]
        classifier_input = torch.cat(list_of_final_embeddings, dim=1)
        logits = self.classifier(classifier_input)
        
        return logits