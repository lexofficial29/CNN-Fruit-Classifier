import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.set_num_threads(os.cpu_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dir = 'train'
test_dir = 'test'
predict_dir = 'predict'


img_height, img_width = 250, 250
batch_size = 32


train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])


train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
predict_dataset = datasets.ImageFolder(predict_dir, transform=test_transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
predict_loader = DataLoader(predict_dataset, batch_size=batch_size)

num_classes = len(train_dataset.classes)

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (img_height // 16) * (img_width // 16), 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNNModel(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


def evaluate(model, loader):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    avg_loss = running_loss / len(loader)
    return accuracy, avg_loss

epochs = 15
for epoch in range(epochs):
    start_time = time.time()

    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_train_loss = running_loss / len(train_loader)

    test_acc, test_loss = evaluate(model, test_loader)
    predict_acc, predict_loss = evaluate(model, predict_loader)

    epoch_time = time.time() - start_time

    print(f"Epoch {epoch+1:02}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f} | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f} | "
          f"Predict Loss: {predict_loss:.4f}, Acc: {predict_acc:.2f} | "
          f"Time: {epoch_time:.2f}s")

    scheduler.step()

torch.save(model.state_dict(), "classifier.pth")
print("Model weights saved as classifier.pth")

