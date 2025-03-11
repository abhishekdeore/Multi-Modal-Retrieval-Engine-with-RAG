import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

class ImageProcessor:
    def __init__(self):
        """
        Initializes the ImageProcessor class by loading the MobileNetV2 model
        with its classifier replaced by an identity function for feature extraction.
        """
        print("Loading image model: MobileNetV2")
        # Load a pre-trained MobileNetV2 model and modify it for feature extraction
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier = torch.nn.Identity()
        self.model.eval()  # Switch the model to evaluation mode to disable dropout, etc.

        # Define a transformation pipeline for preparing images
        self.transform = transforms.Compose([
            transforms.Resize(256),             # Resize the image to 256x256
            transforms.CenterCrop(224),         # Crop the image to 224x224
            transforms.ToTensor(),              # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet means and stds
                                std=[0.229, 0.224, 0.225])
        ])
        print("Image model loaded successfully")
    
    def encode(self, images, batch_size=16):
        """
        Encodes a batch of images into embeddings using the pre-loaded model.

        Args:
        images (list of PIL.Image): A list of image objects to be processed.
        batch_size (int): The number of images to process in a single batch.

        Returns:
        np.array: An array of image embeddings.
        """
        embeddings = []

        # Process images in batches to manage memory usage
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensors = torch.stack([self.transform(img) for img in batch])
            
            with torch.no_grad():  # Disable gradient computation for inference
                batch_embeddings = self.model(batch_tensors).squeeze().numpy()
            
            if len(batch) == 1:
                # Handle single image case which doesn't return a batch dimension
                embeddings.append(batch_embeddings)
            else:
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def process_image_files(self, directory, batch_size=16):
        """
        Processes all image files in a specified directory, encoding their contents into embeddings.

        Args:
        directory (str): The directory to search for image files.
        batch_size (int): The batch size for processing images.

        Returns:
        dict: A dictionary containing embeddings, the original images, and metadata about the files.
        """
        file_paths = []
        images = []
        
        # Collect all image files with valid extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                file_path = os.path.join(directory, filename)
                try:
                    img = Image.open(file_path).convert('RGB')
                    file_paths.append(file_path)
                    images.append(img)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Encode images and log the process
        print(f"Encoding {len(images)} images...")
        embeddings = self.encode(images, batch_size=batch_size)
        
        # Generate metadata for each image processed
        metadata = []
        for i, path in enumerate(file_paths):
            filename = os.path.basename(path)
            metadata.append({
                "id": f"image_{i}",
                "path": path,
                "filename": filename,
                "modality": "image"
            })
        
        return {
            "embeddings": embeddings,
            "images": images,
            "metadata": metadata
        }
