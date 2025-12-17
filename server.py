# server.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image, ImageFilter
import io
import numpy as np
import cv2

app = FastAPI()

# Allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ------------------------
# Load model
# ------------------------
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load('models/waste_classifier_final.pth', map_location='cpu'))
model.eval()

# ------------------------
# Image preprocessing
# ------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------
# Blur detection & sharpening
# ------------------------
def is_blurry_pil(img, threshold=100):
    """Detects if a PIL image is blurry using Laplacian variance"""
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(img_gray, cv2.CV_64F).var() < threshold

def sharpen_image(img, radius=2, percent=150, threshold=3):
    """Applies sharpening to a PIL image"""
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

# ------------------------
# Prediction endpoint
# ------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Read image
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # ------------------------
        # Blur check and sharpening
        # ------------------------
        if is_blurry_pil(img):
            print("Warning: Image may be blurry. Applying sharpening...")
            img = sharpen_image(img)

        # ------------------------
        # Transform and predict
        # ------------------------
        img_t = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            pred_class = class_names[predicted.item()]

        # Low-confidence warning
        if confidence.item() < 0.6:
            print(f"Warning: Low confidence ({confidence.item():.2f}) — prediction may be unreliable.")

        return JSONResponse({
            "prediction": pred_class,
            "confidence": float(confidence.item())
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
