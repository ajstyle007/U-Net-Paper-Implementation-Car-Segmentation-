import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from unet import UNet


# ✅ Set page config must come first
st.set_page_config(page_title="Car Segmentation", page_icon="🚗", layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load your trained model
@st.cache_resource
def load_model():
    # re-create model architecture
    model = UNet().to(device)

    # load saved weights
    checkpoint = torch.load("best_model_new.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    return model

model = load_model()

# ----------------------------
# Define same transform as training
# ----------------------------
img_transform = T.Compose([
    T.Resize((256, 256)),   # match training input size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Streamlit UI
# ----------------------------
# st.set_page_config(page_title="Car Segmentation", page_icon="🚗", layout="wide")
st.title("🚗 Car Segmentation Web App")
st.write("Upload a car image and the model will generate its segmentation mask.")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])


col1, col2, col3 = st.columns(3)


if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    # st.image(image, caption="Uploaded Image", use_container_width=True)
    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    input_tensor = img_transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)  # [1, 1, H, W]
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float()

    # Resize mask back to original size
    h, w = image.size[1], image.size[0]  # (height, width)
    pred_mask_resized = TF.resize(
        TF.to_pil_image(pred_mask.squeeze().cpu()), 
        size=(h, w), 
        interpolation=Image.NEAREST
    )

    # Show results
    
    # st.image(pred_mask_resized, caption="Segmentation Mask", use_container_width=True)
    with col2:
        st.subheader("Predicted Mask")
        st.image(pred_mask_resized, caption="Segmentation Mask", width=300)

    # Overlay (optional)
    mask_np = np.array(pred_mask_resized)
    overlay = np.array(image).copy()
    overlay[mask_np > 128] = [255, 0, 0]  # red mask overlay
    
    # st.image(overlay, caption="Overlay", use_container_width=True)
    with col3:
        st.subheader("Overlay on Original Image")
        st.image(overlay, caption="Overlay", width=300)
