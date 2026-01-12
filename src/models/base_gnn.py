"""
Simple Base GNN for pocket detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SimpleGNN(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, output_dim=1):
        super().__init__()
        
        # Graph attention layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2, dropout=0.1)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, dropout=0.1)
        self.conv3 = GATConv(hidden_dim * 2, hidden_dim, dropout=0.1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GAT layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # Classify each node
        out = self.classifier(x)
        
        return out
    
    def predict(self, data, threshold=0.5):
        """Make predictions with probability threshold"""
        with torch.no_grad():
            logits = self.forward(data)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            return probs, preds

def test_model():
    """Test the GNN model"""
    print("Testing SimpleGNN model...")
    
    # First, try to load a sample graph
    try:
        import os
        sample_dir = "./data/processed/graphs/samples"
        graph_files = [f for f in os.listdir(sample_dir) if f.endswith('.pt')]
        
        if not graph_files:
            print("No sample graphs found. Run graph builder first.")
            return False
        
        # Load first graph
        graph_path = os.path.join(sample_dir, graph_files[0])
        print(f"Loading graph: {graph_path}")
        graph = torch.load(graph_path, weights_only=False)
        
        # Create model
        input_dim = graph.x.shape[1]
        model = SimpleGNN(input_dim=input_dim)
        print(f"Model created with input_dim={input_dim}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(graph)
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Test prediction
        probs, preds = model.predict(graph)
        print(f"Probabilities range: [{probs.min():.3f}, {probs.max():.3f}]")
        print(f"Positive predictions: {preds.sum().item()}/{len(preds)}")
        
        print("\n✓ SimpleGNN test PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ SimpleGNN test FAILED: {e}")
        return False

if __name__ == "__main__":
    test_model()
