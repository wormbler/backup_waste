import torch
from torchvision import models, transforms
from PIL import Image

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

model = models.resnet18()
num_classes = len(class_names)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('models/waste_classifier_final.pth'))
model.eval()

img = Image.open('download.png').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_t)
    _, predicted = torch.max(outputs, 1)
    print("Predicted class index:", predicted.item())
    print("Predicted class name:", class_names[predicted.item()])