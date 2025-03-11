import chromadb
import numpy as np
from tqdm import tqdm
from PIL import Image

class MultiModalRetrievalEngine:
    def __init__(self):
        """
        Initializes the MultiModalRetrievalEngine by setting up connections to ChromaDB collections for
        both text and image data.
        """
        # Establish a persistent client to manage database connections
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Attempt to retrieve or create necessary collections for storing data
        try:
            self.text_collection = self.client.get_or_create_collection("text_data")
            self.image_collection = self.client.get_or_create_collection("image_data")
        except Exception as e:
            print(f"Error initializing collections: {e}")
            # Attempt to create new collections if initial retrieval fails
            try:
                self.text_collection = self.client.create_collection("text_data")
                self.image_collection = self.client.create_collection("image_data")
            except Exception as e:
                print(f"Error creating collections: {e}")
        
        self.text_processor = None
        self.image_processor = None
    
    def set_text_processor(self, processor):
        """Sets the text processor for encoding text data."""
        self.text_processor = processor
    
    def set_image_processor(self, processor):
        """Sets the image processor for encoding image data."""
        self.image_processor = processor
    
    def index_text_data(self, text_data):
        """
        Indexes text data by adding it to the ChromaDB text collection.
        
        Args:
        text_data (dict): A dictionary containing embeddings, metadata, and text data.
        """
        embeddings = text_data["embeddings"]
        metadata = text_data["metadata"]
        texts = text_data["texts"]
        
        # Prepare batches for data insertion to manage memory usage effectively
        ids = [item["id"] for item in metadata]
        metadatas = [item for item in metadata]
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_metadatas = metadatas[i:end_idx]
            batch_documents = texts[i:end_idx]
            
            # Add batch data to the text collection
            self.text_collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )
        
        print(f"Indexed {len(ids)} text documents")
    
    def index_image_data(self, image_data):
        """
        Indexes image data by adding it to the ChromaDB image collection.
        
        Args:
        image_data (dict): A dictionary containing embeddings and metadata.
        """
        embeddings = image_data["embeddings"]
        metadata = image_data["metadata"]
        
        # Prepare batches for data insertion
        ids = [item["id"] for item in metadata]
        metadatas = [item for item in metadata]
        documents = [item["path"] for item in metadata]
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_metadatas = metadatas[i:end_idx]
            batch_documents = documents[i:end_idx]
            
            # Add batch data to the image collection
            self.image_collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )
        
        print(f"Indexed {len(ids)} images")
    
    def search_text(self, query, n_results=5):
        """
        Searches for text documents in the text collection matching a given query.
        
        Args:
        query (str): The text query to search for.
        n_results (int): The number of results to return.
        
        Returns:
        dict: A dictionary containing the search results.
        """
        if self.text_processor is None:
            raise ValueError("Text processor not set")
        
        # Encode the query and perform the search
        query_embedding = self.text_processor.encode([query])[0].tolist()
        results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        print(f"Text search results: {len(results['documents']) if 'documents' in results else 0} documents found")
        return results

    def search_image(self, query_image, n_results=5):
        """
        Searches for images in the image collection similar to a given query image.
        
        Args:
        query_image (Image): The image to use as a search query.
        n_results (int): The number of results to return.
        
        Returns:
        dict: A dictionary containing the search results.
        """
        if self.image_processor is None:
            raise ValueError("Image processor not set")
        
        # Encode the query image and perform the search
        query_embedding = self.image_processor.encode([query_image])[0].tolist()
        results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        print(f"Image search results: {len(results['documents']) if 'documents' in results else 0} documents found")
        return results
