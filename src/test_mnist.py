import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import json
import csv
import os
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Create results directory
results_dir = 'mnist_results'
os.makedirs(results_dir, exist_ok=True)

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Download and load training data
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False
)

# Define the neural network
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Initialize model, loss function, and optimizer
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train_model():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    return avg_loss, accuracy

# Test function
def test_model():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, accuracy

# Save training results to CSV
def save_results_to_csv(train_losses, train_accuracies, test_losses, test_accuracies):
    csv_path = os.path.join(results_dir, 'training_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                f'{train_losses[epoch]:.4f}',
                f'{train_accuracies[epoch]:.2f}',
                f'{test_losses[epoch]:.4f}',
                f'{test_accuracies[epoch]:.2f}'
            ])

# Save final results to JSON
def save_final_results(final_train_acc, final_test_acc, final_train_loss, final_test_loss):
    results = {
        'final_train_accuracy': final_train_acc,
        'final_test_accuracy': final_test_acc,
        'final_train_loss': final_train_loss,
        'final_test_loss': final_test_loss,
        'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs
        }
    }
    
    json_path = os.path.join(results_dir, 'final_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

# Save plots to files
def save_plots(train_losses, train_accuracies, test_losses, test_accuracies):
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.savefig(os.path.join(results_dir, 'loss_plot.png'))
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Test Accuracy')
    plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'))
    plt.close()

# Training loop
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print("Starting training...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_model()
    test_loss, test_acc = test_model()
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    print('-' * 50)

# Save all results
save_results_to_csv(train_losses, train_accuracies, test_losses, test_accuracies)
save_final_results(train_accuracies[-1], test_accuracies[-1], train_losses[-1], test_losses[-1])
save_plots(train_losses, train_accuracies, test_losses, test_accuracies)

# Save the model
model_path = os.path.join(results_dir, 'mnist_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved as '{model_path}'")

# Save model architecture info
model_info = {
    'model_architecture': str(model),
    'total_parameters': sum(p.numel() for p in model.parameters()),
    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
}

with open(os.path.join(results_dir, 'model_info.json'), 'w') as f:
    json.dump(model_info, f, indent=4)

print(f"\nTraining completed! Results saved in '{results_dir}' directory")
print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")