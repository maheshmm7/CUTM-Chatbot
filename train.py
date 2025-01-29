import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # Add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = tokenize(pattern)
        # Add to our words list
        all_words.extend(w)
        # Add to xy pair
        xy.append((w, tag))

# Stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

# Dataset definition
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Store the loss values, precision, and accuracy for plotting
loss_values = []
precision_values = []
accuracy_values = []

# Function to calculate precision and accuracy
def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    precision = correct / len(labels)
    accuracy = correct / labels.size(0)
    return precision, accuracy

# Train the model
for epoch in range(num_epochs):
    epoch_loss = 0.0
    all_labels = []
    all_predicted = []

    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Accumulate all labels and predictions
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    precision = np.mean(np.array(all_labels) == np.array(all_predicted))
    accuracy = precision  # Precision and accuracy are the same in this context
    
    loss_values.append(avg_epoch_loss)
    precision_values.append(precision)
    accuracy_values.append(accuracy)
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f}')

print(f'final loss: {avg_epoch_loss:.4f}')
print(f'final precision: {precision:.4f}')
print(f'final accuracy: {accuracy:.4f}')

# Directory to save figures
figures_dir = "training_figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Directory '{figures_dir}' created")

# First Plot: All metrics in one graph
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Loss')
plt.plot(accuracy_values, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Metrics Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(figures_dir, 'training_metrics_combined.png'))
plt.show()

# Second Plot: Separate subplots for each metric
plt.figure(figsize=(18, 5))
# Plot the loss values
plt.subplot(1, 3, 1)
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')

# Plot the precision values
plt.subplot(1, 3, 2)
plt.plot(precision_values)
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Training Precision Over Epochs')

# Plot the accuracy values
plt.subplot(1, 3, 3)
plt.plot(accuracy_values)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Epochs')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'training_metrics_separate.png'))
plt.show()

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
