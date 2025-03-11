from sentence_transformers import SentenceTransformer
import os
import json
from tqdm import tqdm
import numpy as np

class TextProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the TextProcessor class by loading a specified sentence transformer model.
        
        Args:
        model_name (str): The name of the model to load. Default is 'all-MiniLM-L6-v2'.
        """
        print(f"Loading text model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Text model loaded successfully")
        
    def encode(self, texts, batch_size=16):
        """
        Encodes a list of text strings into embeddings using the loaded model.
        
        Args:
        texts (list of str): A list of texts to encode.
        batch_size (int): The batch size for processing. Default is 16.
        
        Returns:
        np.array: An array of embeddings.
        """
        # Encode the texts and display progress bar
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    def process_text_files(self, directory, batch_size=16):
        """
        Processes all text files in a specified directory, encoding their contents into embeddings.
        
        Args:
        directory (str): The directory to search for text files.
        batch_size (int): The batch size for processing. Default is 16.
        
        Returns:
        dict: A dictionary containing embeddings, original texts, and metadata about the files.
        """
        file_paths = []
        texts = []
        
        # Collect all text files from the directory
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                file_paths.append(file_path)
                texts.append(text)
        
        # Encode the collected texts
        print(f"Encoding {len(texts)} text documents...")
        embeddings = self.encode(texts, batch_size=batch_size)
        
        # Prepare metadata for each file
        metadata = []
        for i, path in enumerate(file_paths):
            filename = os.path.basename(path)
            metadata.append({
                "id": f"text_{i}",
                "path": path,
                "filename": filename,
                "modality": "text"
            })
        
        # Return embeddings, original texts, and their metadata
        return {
            "embeddings": embeddings,
            "texts": texts,
            "metadata": metadata
        }
