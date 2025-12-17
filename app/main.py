from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os

app = FastAPI()

# Load your model
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model = models.resnet18()
num_classes = len(class_names)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Adjust path if needed
model_path = os.path.join(os.path.dirname(__file__), '../models/waste_classifier_final.pth')
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()


# Transform function
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Homepage
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <h2>Waste Classifier</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """


# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)

    return {"predicted_class": class_names[predicted.item()]}
