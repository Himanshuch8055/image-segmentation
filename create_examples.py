import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import math

def create_filament_network(size=(512, 512), num_nodes=15, num_filaments=30):
    """Create a filament network similar to microscopy images"""
    # Create a black background
    img = Image.new('L', size, 0)
    draw = ImageDraw.Draw(img)
    
    # Generate random nodes (points where filaments connect)
    nodes = [(
        random.randint(50, size[0]-50),
        random.randint(50, size[1]-50)
    ) for _ in range(num_nodes)]
    
    # Connect nodes with curved lines to form filaments
    for _ in range(num_filaments):
        if len(nodes) < 2:
            break
            
        # Select two random nodes
        n1, n2 = random.sample(nodes, 2)
        x1, y1 = n1
        x2, y2 = n2
        
        # Create a curved line between nodes
        control_x = (x1 + x2) // 2 + random.randint(-100, 100)
        control_y = (y1 + y2) // 2 + random.randint(-100, 100)
        
        # Draw the curved line with varying thickness
        thickness = random.uniform(0.5, 2.5)
        steps = 50
        for i in range(steps):
            t = i / steps
            # Quadratic bezier curve
            x = int((1-t)**2 * x1 + 2*(1-t)*t * control_x + t**2 * x2)
            y = int((1-t)**2 * y1 + 2*(1-t)*t * control_y + t**2 * y2)
            
            # Vary the brightness along the filament
            brightness = int(100 + 100 * math.sin(t * math.pi))
            
            # Draw a small circle for each point in the curve
            r = int(thickness * random.uniform(0.8, 1.2))
            draw.ellipse((x-r, y-r, x+r, y+r), fill=brightness)
    
    # Add some noise
    img_array = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 5, size).astype(np.float32)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Add some bright dots (particles)
    for _ in range(random.randint(20, 50)):
        x = random.randint(0, size[0]-1)
        y = random.randint(0, size[1]-1)
        r = random.randint(1, 3)
        brightness = random.randint(150, 255)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=brightness)
    
    # Apply slight blur to make it look more natural
    img = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return img

def create_sample_image(size=(512, 512)):
    """Create a sample microscopy-like image with filament networks"""
    # Create multiple layers of filament networks
    img1 = create_filament_network(size, num_nodes=15, num_filaments=30)
    img2 = create_filament_network(size, num_nodes=10, num_filaments=20)
    
    # Combine layers with different opacities
    img = Image.blend(img1, img2, alpha=0.3)
    
    # Add some global noise
    noise = np.random.normal(0, 3, size).astype(np.uint8)
    img_array = np.array(img).astype(np.float32) + noise
    img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    # Adjust contrast
    img = img.point(lambda p: p * 1.2 if p > 30 else p)
    
    return img

def create_examples():
    """Create example images in the examples directory"""
    # Create examples directory if it doesn't exist
    os.makedirs('examples', exist_ok=True)
    
    # Clear existing examples
    for f in os.listdir('examples'):
        if f.startswith('example') and (f.endswith('.jpg') or f.endswith('.png')):
            os.remove(os.path.join('examples', f))
    
    # Create 3 sample images
    for i in range(1, 4):
        img = create_sample_image()
        img_path = f'examples/example{i}.jpg'
        img.save(img_path, 'JPEG', quality=95, subsampling=0)
        print(f'Created {img_path}')

if __name__ == '__main__':
    create_examples()
