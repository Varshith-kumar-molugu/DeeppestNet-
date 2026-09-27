import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# ------------------------------------------------------------
# CBAM Attention Module and DeepPestNet Model 
# (Mirroring the definitions from data.ipynb)
# ------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels//16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels//16, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg = torch.mean(x.view(b, c, -1), dim=2)
        mx = torch.max(x.view(b, c, -1), dim=2)[0]
        att = self.fc2(self.relu(self.fc1(avg))) + self.fc2(self.relu(self.fc1(mx)))
        att = self.sigmoid(att).unsqueeze(2).unsqueeze(3)
        return x * att

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        att = torch.cat([avg, mx], dim=1)
        att = self.sigmoid(self.conv(att))
        return x * att

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class DeepPestNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.backbone = base_model
        self.att = CBAM(1280)
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.att(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.fc(x)

# ------------------------------------------------------------
# FastAPI Setup
# ------------------------------------------------------------
app = FastAPI(title="Crop Pest Detection API")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']
IMG_SIZE = 224

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize the model structure
try:
    base = models.efficientnet_b0(weights=None)
except TypeError: # Older torchvision fallback
    base = models.efficientnet_b0(pretrained=False)

base.classifier = nn.Identity()
model = DeepPestNet(base, len(CLASSES)).to(DEVICE)

# Load the trained weights
model_path = "deeppestnet_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"DeepPestNet Model Loaded from {model_path}")
else:
    print(f"Warning: {model_path} not found in the current directory.")
    print("Continuing with untrained weights for demonstration.")

model.eval()

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_pest(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        tensor = val_tf(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        pest_class = CLASSES[predicted.item()]
        conf_percentage = f"{confidence.item() * 100:.2f}%"
        
        return {
            "prediction": pest_class.capitalize(),
            "confidence": conf_percentage,
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}
