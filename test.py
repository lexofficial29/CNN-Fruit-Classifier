import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from collections import defaultdict

img_height = 250
img_width = 250

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

# ---------- CONFIG ----------
model_path = "classifier.pth"
predict_dir = "predict"
class_names = sorted(os.listdir("predict"))  # assumes 'train/' exists with subfolders
num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((250, 250)), 
    transforms.ToTensor(),
])

model = CNNModel(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


total = defaultdict(int)
correct = defaultdict(int)


for true_class in os.listdir(predict_dir):
    class_folder = os.path.join(predict_dir, true_class)
    if not os.path.isdir(class_folder):
        continue

    for filename in os.listdir(class_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(class_folder, filename)
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                predicted_class = class_names[predicted.item()]

            total[true_class] += 1
            if predicted_class == true_class:
                correct[true_class] += 1

            print(f"[{true_class}] {filename} → Predicted: {predicted_class}")

        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

print("\n--- Classification Report ---")
overall_correct = 0
overall_total = 0
for cls in sorted(total):
    acc = 100.0 * correct[cls] / total[cls] if total[cls] > 0 else 0
    print(f"{cls:<12} | Accuracy: {acc:.2f}% ({correct[cls]}/{total[cls]})")
    overall_correct += correct[cls]
    overall_total += total[cls]

overall_acc = 100.0 * overall_correct / overall_total if overall_total > 0 else 0
print(f"\nOverall Accuracy: {overall_acc:.2f}% ({overall_correct}/{overall_total})")
