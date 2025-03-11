from src.text_processor import TextProcessor
from src.image_processor import ImageProcessor
from src.retrieval_engine import MultiModalRetrievalEngine
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

def main():
    # Initialize the text and image processors along with the retrieval engine
    text_processor = TextProcessor()
    image_processor = ImageProcessor()
    
    engine = MultiModalRetrievalEngine()
    engine.set_text_processor(text_processor)
    engine.set_image_processor(image_processor)
    
    # Process text and image files from specified directories and index them in the retrieval engine
    text_data = text_processor.process_text_files('data/text')
    engine.index_text_data(text_data)
    
    image_data = image_processor.process_image_files('data/images')
    engine.index_image_data(image_data)
    
    # Start a simple command-line interface to interact with the retrieval engine
    while True:
        print("\nMulti-Modal Retrieval Engine")
        print("1. Search for text documents")
        print("2. Search for images using a text query")
        print("3. Search for images using an image")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            query = input("Enter your text query: ")
            results = engine.search_text(query)
            
            print("\nSearch Results:")
            # Display text search results along with a short excerpt from each document
            for i, (doc, meta, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                print(f"{i+1}. {meta['filename']} (Score: {1-distance:.4f})")
                print(f"   Excerpt: {doc[:100]}...\n")
        
        elif choice == '2':
            query = input("Enter your text query for images: ")
            results = engine.search_text_for_images(query)
            
            print("\nSearch Results:")
            # Display image search results and their paths
            for i, (path, meta, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                print(f"{i+1}. {meta['filename']} (Score: {1-distance:.4f})")
                print(f"   Path: {path}\n")
        
        elif choice == '3':
            path = input("Enter the path to your query image: ")
            try:
                query_image = Image.open(path).convert('RGB')
                results = engine.search_image(query_image)
                
                print("\nSearch Results:")
                # Display image search results and their paths
                for i, (path, meta, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    print(f"{i+1}. {meta['filename']} (Score: {1-distance:.4f})")
                    print(f"   Path: {path}\n")
            except Exception as e:
                print(f"Error loading image: {e}")
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
