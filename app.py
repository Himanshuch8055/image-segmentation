import os
import time
import torch
import numpy as np
from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import gradio as gr
from gradio.themes import Soft
import matplotlib.pyplot as plt
from typing import Tuple

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

# ─── Visualization Function ───────────────────────────────
def visualize_prediction(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Create a side-by-side visualization of input and output."""
    # Convert to RGB if grayscale
    if len(np.array(image).shape) == 2:
        image = ImageOps.grayscale(image).convert('RGB')
    
    # Create mask with color overlay (red with transparency)
    mask_np = np.array(mask.convert('L'))
    colored_mask = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
    colored_mask[..., 0] = 255  # Red
    colored_mask[..., 3] = (mask_np > 0) * 128  # 50% opacity
    colored_mask = Image.fromarray(colored_mask)
    
    # Create side-by-side comparison
    width, height = image.size
    result = Image.new('RGB', (width * 2, height))
    result.paste(image, (0, 0))
    result.paste(Image.blend(image, colored_mask.convert('RGB'), 0.3), (width, 0))
    
    return result

# ─── Enhanced Prediction Function ─────────────────────────
def process_image(image: Image.Image, threshold: float = 0.5) -> Tuple[Image.Image, Image.Image, float]:
    """Process image and return original, mask, and processing time."""
    start_time = time.time()
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Process image
    img_np = np.array(image)
    img_tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
        mask = (pred > threshold).float().cpu().squeeze().numpy()
    
    # Create mask image
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    return image, mask_img, process_time

# ─── Main Prediction Function ─────────────────────────────
def predict(image: Image.Image, threshold: float) -> Tuple[Image.Image, str]:
    """Main prediction function with error handling."""
    try:
        if image is None:
            return None, "⚠️ Please upload an image first."
            
        original, mask, proc_time = process_image(image, threshold)
        visualization = visualize_prediction(original, mask)
        
        # Create info text
        info_text = f"✅ Processed in {proc_time:.2f}s | Threshold: {threshold:.2f} | Size: {image.size[0]}×{image.size[1]}"
        
        return visualization, info_text
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ─── Example Images ───────────────────────────────────────
example_images = [
    ["examples/example1.jpg"],
    ["examples/example2.jpg"],
    ["examples/example3.jpg"]
]

# ─── Custom CSS ───────────────────────────────────────────
custom_css = """
#output-image, #input-image {
    margin: 0 auto;
    max-height: 60vh;
    width: 100%;
    border-radius: 8px;
}

.processing-info {
    font-size: 0.9em;
    color: #666;
    margin-top: 10px;
    padding: 8px;
    background: #f8f9fa;
    border-radius: 4px;
}

.upload-box {
    min-height: 200px;
    border: 2px dashed #666;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
}

/* Style for group containers */
.gr-group {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Button styles */
button {
    border-radius: 6px !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .gr-row {
        flex-direction: column;
    }
    
    .gr-column {
        width: 100% !important;
    }
}
"""

# ─── Create Theme ─────────────────────────────────────────
theme = Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    spacing_size="sm",
    radius_size="lg"
)

# ─── Gradio Interface ────────────────────────────────────
with gr.Blocks(theme=theme, css=custom_css) as demo:
    gr.Markdown("""
    # 🖼️ Fibril Segmentation with UNet++
    Upload a microscopy image to generate a segmentation mask. Adjust the confidence threshold to control sensitivity.
    """)
    
    with gr.Row():
        with gr.Column():
            # Input Section with Card-like Styling
            with gr.Group():
                gr.Markdown("### 🖼️ Input Image")
                input_image = gr.Image(
                    type="pil",
                    label="Upload Image",
                    elem_id="input-image",
                    height=300
                )
                
                with gr.Row():
                    threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Higher values = more conservative segmentation"
                    )
                    
                    submit_btn = gr.Button("Segment Image 🚀", variant="primary")
            
            # How to Use Section
            with gr.Accordion("ℹ️ How to use", open=False):
                gr.Markdown("""
                1. Upload a microscopy image or use one of the examples below
                2. Adjust the confidence threshold if needed
                3. Click 'Segment Image' to process
                4. View the results side-by-side (original | segmented)
                """)
            
            # Model Information Section
            with gr.Accordion("📊 Model Information", open=False):
                gr.Markdown(f"""
                - **Model**: UNet++ with ResNet34 encoder
                - **Input Size**: {CONFIG['img_size']}×{CONFIG['img_size']} pixels
                - **Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}
                - **Normalization**: Mean=0.5, Std=0.5
                """)
        
        # Results Section
        with gr.Column():
            with gr.Group():
                gr.Markdown("### 🔍 Segmentation Result")
                output_image = gr.Image(
                    type="pil",
                    label="Segmentation Result",
                    elem_id="output-image",
                    height=400
                )
                
                with gr.Row():
                    download_btn = gr.Button("⬇️ Download Result")
                    clear_btn = gr.Button("🔄 Clear")
                
                info_text = gr.Markdown("", elem_classes="processing-info")
    
    # Example images
    gr.Markdown("### 🧪 Example Images")
    gr.Examples(
        examples=example_images,
        inputs=[input_image],
        outputs=[output_image, info_text],
        fn=predict,
        cache_examples=True,
        label="Click on any example below to try it out!"
    )
    
    # Event handlers
    submit_btn.click(
        fn=predict,
        inputs=[input_image, threshold],
        outputs=[output_image, info_text]
    )
    
    threshold.change(
        fn=predict,
        inputs=[input_image, threshold],
        outputs=[output_image, info_text]
    )
    
    clear_btn.click(
        fn=lambda: [None, None, ""],
        outputs=[input_image, output_image, info_text]
    )

if __name__ == "__main__":
    demo.launch(share=False, show_error=True)

