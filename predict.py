import torch
from torchvision.models import mobilenet_v2
from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load('models/waste_classifier_final.pth'))
model.eval()

img_path = 'download.png'
img = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img_t = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_t)
    _, predicted = torch.max(outputs, 1)
    print("Predicted class:", class_names[predicted.item()])
