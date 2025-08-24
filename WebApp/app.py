import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (assuming UNet is defined in unet.py)
def load_model():
    from unet import UNet
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_car_final.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# Image transforms
img_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    orig = "static/input.jpg" if os.path.exists(os.path.join(UPLOAD_FOLDER, "input.jpg")) else None
    mask = "static/mask.png" if os.path.exists(os.path.join(UPLOAD_FOLDER, "mask.png")) else None
    overlay = "static/overlay.png" if os.path.exists(os.path.join(UPLOAD_FOLDER, "overlay.png")) else None

    if request.method == "POST":
        # Handle image upload
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return "No file selected", 400

            # Save uploaded image
            img_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
            file.save(img_path)
            orig = "static/input.jpg"
            # Clear previous results
            mask = None
            overlay = None
            for path in [os.path.join(UPLOAD_FOLDER, "mask.png"), os.path.join(UPLOAD_FOLDER, "overlay.png")]:
                if os.path.exists(path):
                    os.remove(path)

        # Handle segmentation
        if "segment" in request.form:
            img_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
            if not os.path.exists(img_path):
                return "No image available for segmentation", 400

            image = Image.open(img_path).convert("RGB")
            input_tensor = img_transform(image).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = torch.sigmoid(output)
                pred_mask = (pred_mask > 0.5).float()

            # Resize mask back to original image size
            mask_resized = TF.resize(
                TF.to_pil_image(pred_mask.squeeze().cpu()),
                size=image.size[::-1],
                interpolation=Image.NEAREST
            )

            # Save mask
            mask_path = os.path.join(UPLOAD_FOLDER, "mask.png")
            mask_resized.save(mask_path)

            # Create overlay
            mask_np = np.array(mask_resized)
            overlay = np.array(image).copy()
            overlay[mask_np > 128] = [255, 0, 0]
            overlay_img = Image.fromarray(overlay)
            overlay_path = os.path.join(UPLOAD_FOLDER, "overlay.png")
            overlay_img.save(overlay_path)

            mask = "static/mask.png"
            overlay = "static/overlay.png"

    return render_template("index.html", orig=orig, mask=mask, overlay=overlay)

if __name__ == "__main__":
    app.run(debug=True)