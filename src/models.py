# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from .feature_extractor import TextFeatureExtractor, AudioFeatureExtractor

class GatedCrossModalAttention(nn.Module):
    """
    Gated Cross-Modal Attention Block
    Dynamically fuses text and audio features with gating mechanism
    """
    def __init__(self, text_dim=768, audio_dim=168, hidden_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_key = nn.Linear(audio_dim, hidden_dim)
        self.audio_value = nn.Linear(audio_dim, hidden_dim)
        self.gate_proj = nn.Linear(audio_dim, 1)
        
    def forward(self, text_features, audio_features):
        # Project features
        Q = self.text_proj(text_features)
        K = self.audio_key(audio_features)
        V = self.audio_value(audio_features)
        
        # Compute attention scores
        attention_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)).squeeze(-1)  # [batch, 1]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute attended audio features
        attended_audio = attention_weights * V
        
        # Compute gating mechanism based on audio quality
        gate = torch.sigmoid(self.gate_proj(audio_features))  # [batch, 1]
        
        # Apply gate to attended audio
        gated_audio = gate * attended_audio
        
        return gated_audio, gate

class ContextAwareHierarchicalMultimodalEncoder(nn.Module):
    """
    CAHME: Context-Aware Hierarchical Multimodal Encoder
    """
    def __init__(self, config, text_dim=768, audio_dim=168, hidden_dim=256, num_classes=7):
        super().__init__()
        self.config = config
        self.context_window = 3
        
        # Context encoding branch
        self.context_lstm = nn.LSTM(
            text_dim, hidden_dim, 
            bidirectional=True, 
            batch_first=True, 
            num_layers=1
        )
        
        # Fusion mechanism
        self.gated_attention = GatedCrossModalAttention(text_dim, audio_dim, hidden_dim)
        
        # Classification head
        fusion_dim = text_dim + hidden_dim  # text + gated audio features
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + hidden_dim * 2, 512),  # + context features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, current_features, history_features=None):
        """
        Args:
            current_features: Dict with 'text', 'audio', 'multimodal'
            history_features: List of previous utterance features
        """
        batch_size = current_features['text'].size(0)
        
        # Process current utterance
        text_features = current_features['text']  # [batch, 768]
        audio_features = current_features['audio']  # [batch, 168]
        
        # Gated cross-modal fusion
        gated_audio, gate_values = self.gated_attention(text_features, audio_features)
        current_fusion = torch.cat([text_features, gated_audio], dim=-1)  # [batch, 768 + 256]
        
        # Process context if available
        if history_features is not None and len(history_features) > 0:
            # Stack historical text features
            history_texts = [hf['text'] for hf in history_features[-self.context_window:]]
            if len(history_texts) > 0:
                history_tensor = torch.stack(history_texts, dim=1)  # [batch, seq_len, 768]
                
                # Encode context with LSTM
                context_out, (hidden, _) = self.context_lstm(history_tensor)
                context_features = torch.cat([hidden[0], hidden[1]], dim=-1)  # [batch, hidden_dim*2]
            else:
                context_features = torch.zeros(batch_size, self.context_lstm.hidden_size * 2).to(text_features.device)
        else:
            context_features = torch.zeros(batch_size, self.context_lstm.hidden_size * 2).to(text_features.device)
        
        # Final classification
        combined_features = torch.cat([current_fusion, context_features], dim=-1)
        logits = self.classifier(combined_features)
        
        return {
            'logits': logits,
            'gate_values': gate_values,
            'context_features': context_features
        }

class DynamicTemporalGraphLayer(nn.Module):
    """
    Single layer of Dynamic Temporal Graph Network
    """
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Graph attention components
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Temporal convolution
        self.temporal_conv = nn.Conv1d(
            input_dim, hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        
        # Edge weight projection
        self.edge_proj = nn.Linear(1, input_dim)  # Project edge weights to feature dimension
        
        # Fusion and normalization
        self.fusion_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, node_features, edge_weights, attention_mask=None):
        """
        Args:
            node_features: [batch, seq_len, input_dim]
            edge_weights: [batch, seq_len, seq_len]
            attention_mask: [batch, seq_len, seq_len]
        """
        batch_size, seq_len, _ = node_features.shape
        
        # Apply edge weights before attention by modifying the attention computation
        if edge_weights is not None:
            # Use edge weights as additional bias in attention
            # Expand edge weights to match attention dimension
            edge_bias = edge_weights.unsqueeze(-1)
            edge_bias = edge_bias.expand(-1, -1, -1, self.input_dim)
        else:
            edge_bias = None
        
        # Convert attention mask format for MultiheadAttention
        if attention_mask is not None:
            attn_mask = attention_mask
            if attn_mask.dim() == 3:
                # Expand mask for multi-head attention
                attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        else:
            attn_mask = None
        
        # Graph Attention
        attended_features, attention_weights = self.attention(
            query=node_features,
            key=node_features,
            value=node_features,
            attn_mask=attn_mask
        )
        
        # Apply edge weights as post-processing (element-wise multiplication)
        if edge_weights is not None:
            # We need to apply edge weights per node
            # For each node, multiply its features by the average edge weight to its neighbors
            edge_weights_normalized = F.softmax(edge_weights, dim=-1)  # Normalize edge weights
            attended_features = torch.bmm(edge_weights_normalized, attended_features)
        
        # Temporal convolution
        temporal_features = node_features.transpose(1, 2)
        temporal_features = self.temporal_conv(temporal_features)
        temporal_features = temporal_features.transpose(1, 2)
        
        # Fusion
        combined = torch.cat([attended_features, temporal_features], dim=-1)
        output = self.fusion_proj(combined)
        output = self.layer_norm(output)
        
        return output, attention_weights

class DynamicTemporalGraphNetwork(nn.Module):
    """
    DTGN: Dynamic Temporal Graph Network for conversation modeling
    """
    def __init__(self, config, input_dim=936, hidden_dim=256, num_layers=3, num_classes=7):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph layers
        self.graph_layers = nn.ModuleList([
            DynamicTemporalGraphLayer(
                hidden_dim, 
                hidden_dim
            ) for _ in range(num_layers)
        ])
        
        # Output classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def build_temporal_edges(self, utterance_ids, speakers, max_time_gap=5):
        """
        Build dynamic edge weights based on temporal proximity and speaker identity
        """
        batch_size, seq_len = utterance_ids.shape
        
        # Create temporal distance matrix
        utterance_expanded = utterance_ids.unsqueeze(-1).expand(-1, -1, seq_len)
        utterance_transposed = utterance_ids.unsqueeze(1).expand(-1, seq_len, -1)
        time_gaps = torch.abs(utterance_expanded - utterance_transposed)
        
        # Convert time gaps to weights (closer = higher weight)
        temporal_weights = 1.0 / (1.0 + torch.log(time_gaps.float() + 1.0))
        
        # Create speaker identity matrix
        speaker_expanded = speakers.unsqueeze(-1).expand(-1, -1, seq_len)
        speaker_transposed = speakers.unsqueeze(1).expand(-1, seq_len, -1)
        speaker_mask = (speaker_expanded == speaker_transposed).float()
        
        # Combine temporal and speaker weights
        edge_weights = temporal_weights * (1.0 + 0.5 * speaker_mask)  # Boost same-speaker edges
        
        # Apply max time gap threshold
        edge_weights = edge_weights * (time_gaps <= max_time_gap).float()
        
        return edge_weights
    
    def forward(self, multimodal_features, utterance_ids, speakers, dialogue_lengths):
        """
        Args:
            multimodal_features: [batch, max_seq_len, input_dim]
            utterance_ids: [batch, max_seq_len] 
            speakers: [batch, max_seq_len] (encoded speaker IDs)
            dialogue_lengths: [batch] actual lengths of each dialogue
        """
        batch_size, max_seq_len, _ = multimodal_features.shape
        
        # Project input features
        node_features = self.input_proj(multimodal_features)  # [batch, seq_len, hidden_dim]
        
        # Build dynamic edges
        edge_weights = self.build_temporal_edges(utterance_ids, speakers)
        
        # Create attention mask for padding
        attention_mask = self._create_attention_mask(dialogue_lengths, max_seq_len)
        
        # Apply graph layers
        attention_weights = []
        for layer in self.graph_layers:
            node_features, attn_weights = layer(
                node_features, 
                edge_weights, 
                attention_mask
            )
            attention_weights.append(attn_weights)
        
        # Classification - use last layer features for each utterance
        logits = self.classifier(node_features)
        
        return {
            'logits': logits,
            'node_features': node_features,
            'attention_weights': attention_weights,
            'edge_weights': edge_weights
        }
    
    def _create_attention_mask(self, dialogue_lengths, max_seq_len):
        """
        Create attention mask to handle variable-length sequences
        """
        batch_size = len(dialogue_lengths)
        mask = torch.ones(batch_size, max_seq_len, max_seq_len)
        
        for i, length in enumerate(dialogue_lengths):
            mask[i, :, length:] = 0  # Mask out padding positions
            mask[i, length:, :] = 0
            
        return mask.bool()

class MultimodalERC(nn.Module):
    """
    Main wrapper class that can use either CAHME or DTGN architecture
    """
    def __init__(self, config, architecture='cahme', num_classes=7):
        super().__init__()
        self.config = config
        self.architecture = architecture
        self.num_classes = num_classes
        
        if architecture.lower() == 'cahme':
            self.model = ContextAwareHierarchicalMultimodalEncoder(
                config, num_classes=num_classes
            )
        elif architecture.lower() == 'dtgn':
            self.model = DynamicTemporalGraphNetwork(
                config, num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def get_architecture_name(self):
        return self.architecture.upper()