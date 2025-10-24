# test_m3fnet_with_real_features.py
import os
import sys
import pickle
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from src.models import MultimodalERC

def test_m3fnet_with_real_features():
    """Test M3F-Net with real extracted features"""
    print("🧪 TESTING M3F-NET WITH REAL FEATURES")
    print("=" * 60)
    
    config = Config()
    
    # Load features
    features_path = os.path.join(config.FEATURES_DIR, "train_features.pkl")
    
    if not os.path.exists(features_path):
        print(f"❌ Features file not found: {features_path}")
        print("💡 Please run feature extraction first:")
        print("   python extract_features.py --split train")
        return
    
    print(f"📂 Loading features from: {features_path}")
    
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    print(f"✅ Successfully loaded features!")
    print(f"   Text features: {features['text_features'].shape}")
    print(f"   Audio features: {features['audio_features'].shape}")
    print(f"   Multimodal features: {features['multimodal_features'].shape}")
    print(f"   Samples: {len(features['emotions'])}")
    
    # Test with a small batch
    num_test_samples = min(8, len(features['emotions']))
    print(f"\n🔧 Testing with {num_test_samples} samples...")
    
    try:
        # Initialize M3F-Net
        m3fnet_model = MultimodalERC(config, architecture='m3fnet', num_classes=len(config.EMOTION_LABELS))
        
        print(f"📊 M3F-Net Architecture:")
        print(f"   - Text encoder: MultiScaleTemporalEncoder")
        print(f"   - Audio encoder: MultiScaleTemporalEncoder") 
        print(f"   - Dynamic feature gating: Quality-aware fusion")
        print(f"   - Cross-modal attention: Multi-head attention")
        print(f"   - Context modeling: Bidirectional LSTM")
        print(f"   - Dropout: Traditional fixed dropout")
        
        # Convert features to tensors
        multimodal_features = torch.tensor(features['multimodal_features'][:num_test_samples]).float()
        
        print(f"\n🎯 Input shapes:")
        print(f"   Multimodal features: {multimodal_features.shape}")
        
        # Check for NaN values
        if torch.isnan(multimodal_features).any():
            print("⚠️  NaN detected in input, applying correction...")
            multimodal_features = torch.nan_to_num(multimodal_features, nan=0.0)
        
        # Forward pass
        print(f"\n🚀 Running forward pass...")
        with torch.no_grad():
            output = m3fnet_model(
                multimodal_features=multimodal_features.unsqueeze(1)  # Add sequence dimension
            )
        
        print(f"✅ M3F-Net forward pass successful!")
        print(f"📈 Output details:")
        print(f"   - Logits shape: {output['logits'].shape}")
        print(f"   - Text confidence: {output['text_confidence'].shape}")
        print(f"   - Audio confidence: {output['audio_confidence'].shape}")
        
        # Show predictions
        print(f"\n🎭 Predictions:")
        probabilities = torch.softmax(output['logits'], dim=1)
        
        for i in range(min(3, num_test_samples)):
            pred_class = torch.argmax(probabilities[i]).item()
            confidence = probabilities[i][pred_class].item()
            true_emotion = features['emotions'][i]
            pred_emotion = config.EMOTION_LABELS[pred_class]
            
            print(f"   Sample {i+1}: True='{true_emotion}', Pred='{pred_emotion}' ({confidence:.3f})")
        
        # Show confidence scores
        print(f"\n🎚️  Confidence Scores (Quality Estimation):")
        for i in range(min(3, num_test_samples)):
            text_conf = output['text_confidence'][i].mean().item()
            audio_conf = output['audio_confidence'][i].mean().item()
            print(f"   Sample {i+1}: Text={text_conf:.3f}, Audio={audio_conf:.3f}")
        
        # Model statistics
        total_params = sum(p.numel() for p in m3fnet_model.parameters())
        trainable_params = sum(p.numel() for p in m3fnet_model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Parameter size: {total_params * 4 / (1024**2):.2f} MB (float32)")
        
    except Exception as e:
        print(f"❌ M3F-Net test failed: {e}")
        import traceback
        traceback.print_exc()

def test_m3fnet_with_dialogue_sequences():
    """Test M3F-Net with proper dialogue sequences"""
    print("\n\n🧪 TESTING M3F-NET WITH DIALOGUE SEQUENCES")
    print("=" * 60)
    
    config = Config()
    
    # Load features
    features_path = os.path.join(config.FEATURES_DIR, "train_features.pkl")
    
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    # Organize into dialogue sequences (similar to DTGN test)
    def create_dialogue_sequences(features, max_dialogues=2):
        dialogues = {}
        
        # Group by dialogue_id
        for i in range(len(features['dialogue_ids'])):
            dia_id = features['dialogue_ids'][i]
            if dia_id not in dialogues:
                dialogues[dia_id] = []
            dialogues[dia_id].append({
                'utterance_id': features['utterance_ids'][i],
                'multimodal_features': features['multimodal_features'][i],
                'emotion': features['emotions'][i]
            })
        
        # Take first N dialogues and sort by utterance_id
        selected_dialogues = list(dialogues.items())[:max_dialogues]
        sequences = []
        dialogue_lengths = []
        
        for dia_id, utterances in selected_dialogues:
            utterances.sort(key=lambda x: x['utterance_id'])
            multimodal_seq = [utt['multimodal_features'] for utt in utterances]
            sequences.append(multimodal_seq)
            dialogue_lengths.append(len(utterances))
        
        # Pad sequences
        max_seq_len = max(dialogue_lengths)
        padded_sequences = []
        
        for seq in sequences:
            pad_length = max_seq_len - len(seq)
            if pad_length > 0:
                padded_seq = np.pad(seq, ((0, pad_length), (0, 0)), mode='constant')
            else:
                padded_seq = np.array(seq)
            padded_sequences.append(padded_seq)
        
        return np.array(padded_sequences), dialogue_lengths
    
    try:
        # Create dialogue sequences
        dialogue_sequences, dialogue_lengths = create_dialogue_sequences(features, max_dialogues=2)
        
        print(f"📊 Dialogue sequence shapes:")
        print(f"   Batch size: {dialogue_sequences.shape[0]} dialogues")
        print(f"   Sequence length: {dialogue_sequences.shape[1]} utterances")
        print(f"   Feature dimension: {dialogue_sequences.shape[2]}")
        print(f"   Dialogue lengths: {dialogue_lengths}")
        
        # Initialize M3F-Net
        m3fnet_model = MultimodalERC(config, architecture='m3fnet', num_classes=len(config.EMOTION_LABELS))
        
        # Convert to tensor
        multimodal_sequences = torch.tensor(dialogue_sequences).float()
        
        # Forward pass with sequences
        print(f"\n🚀 Running forward pass with dialogue sequences...")
        with torch.no_grad():
            output = m3fnet_model(multimodal_features=multimodal_sequences)
        
        print(f"✅ M3F-Net sequence processing successful!")
        print(f"📈 Output details:")
        print(f"   - Logits shape: {output['logits'].shape}")
        print(f"   - Expected: ({multimodal_sequences.shape[0]}, {len(config.EMOTION_LABELS)})")
        
        # Show predictions per dialogue
        print(f"\n🎭 Predictions per dialogue:")
        probabilities = torch.softmax(output['logits'], dim=1)
        
        for i in range(len(dialogue_lengths)):
            print(f"   Dialogue {i+1} ({dialogue_lengths[i]} utterances):")
            pred_class = torch.argmax(probabilities[i]).item()
            confidence = probabilities[i][pred_class].item()
            pred_emotion = config.EMOTION_LABELS[pred_class]
            print(f"     Overall prediction: '{pred_emotion}' ({confidence:.3f})")
            
    except Exception as e:
        print(f"❌ M3F-Net dialogue test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all M3F-Net tests"""
    print("🚀 COMPREHENSIVE M3F-NET TESTING")
    print("=" * 60)
    
    # Test 1: Single utterance processing
    test_m3fnet_with_real_features()
    
    # Test 2: Dialogue sequence processing  
    test_m3fnet_with_dialogue_sequences()
    
    print("\n" + "=" * 60)
    print("✅ M3F-NET TESTING COMPLETED!")

if __name__ == "__main__":
    main()