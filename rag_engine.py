from src.rag_processor import RAGProcessor
import os

class RAGEngine:
    def __init__(self, retrieval_engine):
        """
        Initializes the RAGEngine with a retrieval engine and a RAGProcessor for generating answers.
        
        Args:
        retrieval_engine (object): An instance of a retrieval engine capable of text and image search.
        """
        self.retrieval_engine = retrieval_engine
        self.rag_processor = RAGProcessor()
        self.initialized = False
        
        # Check for an API key in the environment variables and initialize if present
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.initialize(api_key)
    
    def initialize(self, api_key):
        """
        Initializes the RAG engine with the specified API key.

        Args:
        api_key (str): API key for the Gemini API.

        Returns:
        bool: True if initialization was successful, otherwise False.
        """
        success = self.rag_processor.initialize(api_key)
        self.initialized = success
        return success
    
    def answer_query(self, query, n_results=3, stream=False):
        """
        Processes and answers a text query using Retrieval-Augmented Generation (RAG).
        
        Args:
        query (str): The user's query to answer.
        n_results (int): Number of relevant documents to retrieve for context.
        stream (bool): Whether to stream the response.

        Returns:
        str: The generated answer or an error message.
        """
        if not self.initialized:
            return "RAG engine not initialized. Please provide a valid API key."
        
        # Retrieve relevant text documents based on the query
        text_results = self.retrieval_engine.search_text(query, n_results)
        
        # Compile the context from retrieved documents
        context = self._extract_context(text_results)
        
        # Use RAG to generate an answer based on the context
        response = self.rag_processor.generate_answer(query, context, stream=stream)
        
        return response
    
    def answer_image_query(self, query, n_results=3, stream=False):
        """
        Processes and answers a query related to images using Retrieval-Augmented Generation (RAG).

        Args:
        query (str): The user's query related to images.
        n_results (int): Number of images to retrieve for context.
        stream (bool): Whether to stream the response.

        Returns:
        tuple: A tuple containing the generated response and image search results.
        """
        if not self.initialized:
            return ("RAG engine not initialized. Please provide a valid API key.", None)
        
        # Search for relevant images based on the text query
        image_results = self.retrieval_engine.search_text_for_images(query, n_results)
        
        # Compile metadata about the images into context
        context = self._extract_image_context(image_results)
        
        # Use RAG to generate an answer based on the image context
        response = self.rag_processor.generate_answer(query, context, stream=stream)
        
        return response, image_results
    
    def _extract_context(self, results):
        """
        Extracts and formats context from text search results.

        Args:
        results (dict): The search results containing documents and metadata.

        Returns:
        str: A formatted string containing the extracted context.
        """
        context = ""
        
        # Format document text and metadata into a readable context
        if 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                context += f"\n--- Document {i+1}: {metadata.get('filename', 'Unknown')} ---\n"
                context += doc[:5000]  # Limit document content to prevent excessive length
                context += "\n\n"
        
        return context
    
    def _extract_image_context(self, results):
        """
        Extracts and formats context from image search results.

        Args:
        results (dict): The search results containing image paths and metadata.

        Returns:
        str: A formatted string containing the extracted context about the images.
        """
        context = "The following images were found in the system:\n\n"
        
        # Format image metadata into a readable context
        if 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
            for i, path in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                context += f"Image {i+1}: {metadata.get('filename', 'Unknown')}\n"
                context += f"Path: {path}\n"
                context += f"Type: {metadata.get('modality', 'Unknown')}\n\n"
        
        return context
