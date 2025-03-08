import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import json
import os

class MNISTMiniModel(nn.Module):
    def __init__(self):
        super(MNISTMiniModel, self).__init__()
        # Much smaller model
        self.fc1 = nn.Linear(400, 32)  
        self.fc2 = nn.Linear(32, 16)  
        self.fc3 = nn.Linear(16, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 400)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def resize_and_center_mnist(img):
    # Resize from 28x28 to 20x20
    transform = transforms.Compose([
        transforms.Resize((20, 20)),
        transforms.ToTensor()
    ])
    return transform(img)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Custom transform to resize images to a 20x20 image
    # IMPORTANT: Removed the normalization to simplify on-chain inference
    transform = transforms.Compose([
        transforms.Resize((20, 20)),
        transforms.ToTensor()
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model = MNISTMiniModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Test
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        print(f'Epoch: {epoch+1}/{epochs}\tAccuracy: {100. * correct / len(test_loader.dataset):.2f}%')
    
    os.makedirs('./models', exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), './models/mnist_mini_model.pth')
    print("Model saved to models directory!")
    
    # Quantize and export weights
    quantize_and_export_weights(model)

def quantize_and_export_weights(model):
    # Get weights and biases - IMPORTANT: TRANSPOSE the weight matrices
    # to match Solidity indexing order
    w1 = model.fc1.weight.data.cpu().numpy().T  # TRANSPOSED (400, 32)
    b1 = model.fc1.bias.data.cpu().numpy()
    w2 = model.fc2.weight.data.cpu().numpy().T  # TRANSPOSED (32, 16)
    b2 = model.fc2.bias.data.cpu().numpy()
    w3 = model.fc3.weight.data.cpu().numpy().T  # TRANSPOSED (16, 10)
    b3 = model.fc3.bias.data.cpu().numpy()
    
    print(f"Weight shapes after transpose:")
    print(f"W1: {w1.shape}, B1: {b1.shape}")
    print(f"W2: {w2.shape}, B2: {b2.shape}")
    print(f"W3: {w3.shape}, B3: {b3.shape}")
    
    # Calculate scale factor
    max_abs_value = max(
        np.max(np.abs(w1)),
        np.max(np.abs(b1)),
        np.max(np.abs(w2)),
        np.max(np.abs(b2)),
        np.max(np.abs(w3)),
        np.max(np.abs(b3))
    )
    
    # Scale factor to convert to int8 range
    scale_factor = 127 / max_abs_value
    
    # Quantize weights and biases
    q_w1 = np.clip(np.round(w1 * scale_factor), -127, 127).astype(np.int8)
    q_b1 = np.clip(np.round(b1 * scale_factor), -127, 127).astype(np.int8)
    q_w2 = np.clip(np.round(w2 * scale_factor), -127, 127).astype(np.int8)
    q_b2 = np.clip(np.round(b2 * scale_factor), -127, 127).astype(np.int8)
    q_w3 = np.clip(np.round(w3 * scale_factor), -127, 127).astype(np.int8)
    q_b3 = np.clip(np.round(b3 * scale_factor), -127, 127).astype(np.int8)
    
    # Use much smaller chunks to stay within gas limits
    chunk_size = 500  # Reduced from 2000
    
    # W1 is 400x32 = 12,800 elements
    w1_chunks = []
    for i in range(0, q_w1.size, chunk_size):
        chunk = q_w1.flatten()[i:i+chunk_size].tolist()
        w1_chunks.append(chunk)
    
    # W2 is 32x16 = 512 elements
    w2_chunks = []
    for i in range(0, q_w2.size, chunk_size):
        chunk = q_w2.flatten()[i:i+chunk_size].tolist()
        w2_chunks.append(chunk)
    
    # W3 is 16x10 = 160 elements
    w3_chunks = []
    for i in range(0, q_w3.size, chunk_size):
        chunk = q_w3.flatten()[i:i+chunk_size].tolist()
        w3_chunks.append(chunk)    
    
    model_data = {
        'scale_factor': int(scale_factor),
        'input_size': 400,
        'hidden1_size': 32,  
        'hidden2_size': 16,  
        'output_size': 10,
        'b1': q_b1.tolist(),
        'b2': q_b2.tolist(), 
        'b3': q_b3.tolist(),
        'w1_chunks': w1_chunks,
        'w2_chunks': w2_chunks,
        'w3_chunks': w3_chunks
    }
    
    os.makedirs('./models', exist_ok=True)
    
    # Export 
    with open('./models/quantized_model_mini.json', 'w') as f:
        json.dump(model_data, f)
    
    print(f"Quantized model exported to models directory with scale factor: {scale_factor}")
    print(f"W1 chunks: {len(w1_chunks)}")
    print(f"W2 chunks: {len(w2_chunks)}")
    print(f"W3 chunks: {len(w3_chunks)}")
    
if __name__ == "__main__":
    train_model()