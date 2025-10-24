# src/models.py (FIXED VERSION)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

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
        attention_scores = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute attended audio features
        attended_audio = attention_weights * V
        
        # Compute gating mechanism based on audio quality
        gate = torch.sigmoid(self.gate_proj(audio_features))
        
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
        fusion_dim = text_dim + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, current_features, history_features=None):
        batch_size = current_features['text'].size(0)
        
        # Process current utterance
        text_features = current_features['text']
        audio_features = current_features['audio']
        
        # Gated cross-modal fusion
        gated_audio, gate_values = self.gated_attention(text_features, audio_features)
        current_fusion = torch.cat([text_features, gated_audio], dim=-1)
        
        # Process context if available
        if history_features is not None and len(history_features) > 0:
            history_texts = [hf['text'] for hf in history_features[-self.context_window:]]
            if len(history_texts) > 0:
                history_tensor = torch.stack(history_texts, dim=1)
                context_out, (hidden, _) = self.context_lstm(history_tensor)
                context_features = torch.cat([hidden[0], hidden[1]], dim=-1)
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

class MultiScaleTemporalEncoder(nn.Module):
    """
    Captures temporal dependencies at multiple scales using parallel convolutions
    """
    def __init__(self, input_dim, hidden_dim=256, scales=[1, 3, 5]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=scale, padding=scale//2)
            for scale in scales
        ])
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x_t = x.transpose(1, 2)
        
        scale_features = []
        for conv in self.convs:
            conv_out = conv(x_t)
            conv_out = conv_out.transpose(1, 2)
            scale_features.append(conv_out)
        
        multi_scale = torch.stack(scale_features, dim=-1).mean(dim=-1)
        attended, _ = self.attention(multi_scale, multi_scale, multi_scale)
        output = self.layer_norm(attended + multi_scale)
        return output

class DynamicFeatureGating(nn.Module):
    """
    Novel feature gating mechanism that adapts to input quality
    """
    def __init__(self, feature_dim, gate_dim=64):
        super().__init__()
        self.quality_estimator = nn.Sequential(
            nn.Linear(feature_dim, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, 1),
            nn.Sigmoid()
        )
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh()
        )
        
    def forward(self, features):
        # Estimate feature quality (confidence score)
        quality_scores = self.quality_estimator(features)
        
        # Enhance features
        enhanced_features = self.feature_enhancer(features)
        
        # Dynamic gating: blend original and enhanced features based on quality
        gated_output = quality_scores * enhanced_features + (1 - quality_scores) * features
        
        return gated_output, quality_scores

class M3FNet(nn.Module):
    def __init__(self, config, text_dim=768, audio_dim=168, hidden_dim=256, num_classes=7):
        super().__init__()
        self.config = config
        
        # Text pathway
        self.text_encoder = MultiScaleTemporalEncoder(text_dim, hidden_dim)
        self.text_gating = DynamicFeatureGating(hidden_dim)
        
        # Audio pathway  
        self.audio_encoder = MultiScaleTemporalEncoder(audio_dim, hidden_dim)
        self.audio_gating = DynamicFeatureGating(hidden_dim)
        
        # Cross-modal fusion
        self.cross_modal_attention = nn.MultiheadAttention(
            hidden_dim * 2, 
            num_heads=config.M3FNET_ATTENTION_HEADS, 
            batch_first=True
        )
        self.cross_modal_dropout = nn.Dropout(config.M3FNET_ATTENTION_DROPOUT)
        
        # Context modeling - FIXED LSTM (proper dropout)
        self.context_lstm = nn.LSTM(
            hidden_dim * 2, 
            hidden_dim, 
            bidirectional=True, 
            batch_first=True,
            num_layers=config.M3FNET_LSTM_LAYERS,
            dropout=config.M3FNET_LSTM_DROPOUT
        )
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(config.M3FNET_CLASSIFIER_DROPOUT_1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.M3FNET_CLASSIFIER_DROPOUT_2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, multimodal_features, utterance_ids=None, speakers=None, dialogue_lengths=None):
        # Handle both single utterances and sequences
        if multimodal_features.dim() == 2:
            # Single utterance: (batch, features) -> add sequence dimension
            multimodal_features = multimodal_features.unsqueeze(1)
        
        batch_size, seq_len, feature_dim = multimodal_features.shape
        
        # Step 1: Split multimodal features
        text_features = multimodal_features[:, :, :768]      # First 768D for text
        audio_features = multimodal_features[:, :, 768:768+168]  # Next 168D for audio
        
        # Step 2: Text pathway processing
        text_encoded = self.text_encoder(text_features)
        text_gated, text_confidence = self.text_gating(text_encoded)
        
        # Step 3: Audio pathway processing  
        audio_encoded = self.audio_encoder(audio_features)
        audio_gated, audio_confidence = self.audio_gating(audio_encoded)
        
        # Step 4: Early fusion
        fused_features = torch.cat([text_gated, audio_gated], dim=-1)
        
        # Step 5: Cross-modal attention with dropout
        cross_modal, attention_weights = self.cross_modal_attention(
            fused_features, fused_features, fused_features
        )
        cross_modal = self.cross_modal_dropout(cross_modal)
        
        # Step 6: Context modeling with LSTM
        context_out, (hidden, _) = self.context_lstm(cross_modal)
        context_features = torch.cat([hidden[0], hidden[1]], dim=-1)
        
        # Step 7: Sequence aggregation (mean pooling)
        sequence_representation = cross_modal.mean(dim=1)
        
        # Step 8: Final classification
        combined_features = torch.cat([sequence_representation, context_features], dim=-1)
        logits = self.classifier(combined_features)
        
        return {
            'logits': logits,
            'text_confidence': text_confidence,
            'audio_confidence': audio_confidence,
            'attention_weights': attention_weights
        }

# Update the main model wrapper to include M3F-Net
class MultimodalERC(nn.Module):
    def __init__(self, config, architecture='cahme', num_classes=7):
        super().__init__()
        self.config = config
        self.architecture = architecture
        self.num_classes = num_classes
        
        if architecture.lower() == 'cahme':
            self.model = ContextAwareHierarchicalMultimodalEncoder(
                config, num_classes=num_classes
            )
        elif architecture.lower() == 'm3fnet':
            self.model = M3FNet(
                config, num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def get_architecture_name(self):
        return self.architecture.upper()