import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import gradio as gr

# ─── Configuration ─────────────────────────────────────────
CONFIG = {
    "model_path": "./model/encoder_resnet34_decoder_UnetPlusPlus_fibril_seg_model.pth",
    "img_size": 512
}

# ─── Device Setup ──────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ─── Load Model ────────────────────────────────────────────
model = smp.UnetPlusPlus(
    encoder_name='resnet34',
    encoder_depth=5,
    encoder_weights='imagenet',
    decoder_use_norm='batchnorm',
    decoder_channels=(256, 128, 64, 32, 16),
    decoder_attention_type=None,
    decoder_interpolation='nearest',
    in_channels=1,
    classes=1,
    activation=None
).to(device)

model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
model.eval()

# ─── Transform Function ────────────────────────────────────
def get_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

transform = get_transform(CONFIG["img_size"])

# ─── Prediction Function ───────────────────────────────────
def predict(image):
    image = image.convert("L")  # Convert to grayscale
    img_np = np.array(image)
    img_tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
        mask = (pred > 0.5).float().cpu().squeeze().numpy()

    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_img

# ─── Gradio Interface ──────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Microscopy Image"),
    outputs=gr.Image(type="pil", label="Predicted Segmentation Mask"),
    title="Fibril Segmentation with Unet++",
    description="Upload a grayscale microscopy image to get its predicted segmentation mask."
)

if __name__ == "__main__":
    demo.launch()
