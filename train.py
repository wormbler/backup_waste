import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
import cv2
import os
import numpy as np

# ------------------------
# Classes
# ------------------------
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
num_classes = len(class_names)

# ------------------------
# Device
# ------------------------
device = torch.device("cpu")  # CPU-only

# ------------------------
# Data transforms
# ------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # blur augmentation
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# ------------------------
# Load datasets
# ------------------------
dataset_path = 'dataset'
train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ------------------------
# Load MobileNetV2
# ------------------------
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# ------------------------
# Freeze feature layers initially
# ------------------------
for param in model.features.parameters():
    param.requires_grad = False

# ------------------------
# Loss, optimizer, scheduler
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

# ------------------------
# Training function
# ------------------------
best_acc = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    global best_acc
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        val_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")
        for i, cls in enumerate(class_names):
            print(f"  {cls}: {100 * class_correct[i] / class_total[i]:.2f}%")

        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_acc:
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/waste_classifier_best.pth')
            best_acc = val_acc
            print("Saved best model.")

# ------------------------
# Phase 1: Train classifier head
# ------------------------
print("=== Phase 1: Training classifier head ===")
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=5)

# ------------------------
# Phase 2: Fine-tune top layers
# ------------------------
print("=== Phase 2: Fine-tuning top layers ===")
for param in model.features[-2:].parameters():  # unfreeze last two layers
    param.requires_grad = True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=5)

print("Training complete. Best model saved as 'models/waste_classifier_best.pth'")

# ------------------------
# Inference helper functions
# ------------------------

# Preprocessing transform for inference
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Optional sharpening for blurry images
def sharpen_image(img, apply_sharpen=True):
    if apply_sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return img

# Blur check function
def is_blurry(image_path, threshold=100):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold

# Prediction function with confidence
def predict_image(image_path, apply_sharpen=False, blur_warning=True):
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Blur detection
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blurry = cv2.Laplacian(img_gray, cv2.CV_64F).var() < 100
    if blurry and blur_warning:
        print("Warning: Image may be too blurry for accurate prediction.")

    # Apply sharpening if requested
    img = sharpen_image(img, apply_sharpen)

    # Preprocess
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    predicted_class = class_names[pred.item()]

    # Low-confidence warning
    if confidence.item() < 0.6:
        print(f"Warning: Model confidence is low ({confidence.item():.2f}) — prediction may be unreliable.")

    return predicted_class, float(confidence.item())

# Example usage
if __name__ == '__main__':
    test_image_path = 'uploads/test_image.jpg'  # replace with your uploaded image path
    prediction, conf = predict_image(test_image_path, apply_sharpen=True, blur_warning=True)
    print(f"Predicted class: {prediction} | Confidence: {conf:.2f}")
