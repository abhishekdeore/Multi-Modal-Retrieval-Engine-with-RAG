# Multi-Modal Retrieval Engine with RAG

A sophisticated search and question-answering system that works across different content types (text and images) with intelligent response generation powered by Google's Gemini API.

## Overview

This project implements a powerful multi-modal retrieval engine with integrated Retrieval-Augmented Generation (RAG). The system can process and retrieve information from various data formats (text and images) and provide intelligent answers to natural language questions by selecting appropriate encoders for each modality.

The application features:
- Text document search using semantic understanding
- Text-to-image search capabilities
- Image-to-image similarity search
- Intelligent question answering using retrieved documents

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Current Limitations](#current-limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Features

### Multi-Modal Search
- **Text Search**: Find semantically relevant text documents based on natural language queries
- **Text-to-Image Search**: Retrieve images using text descriptions
- **Image-to-Image Search**: Find visually similar images by uploading a query image

### Retrieval-Augmented Generation (RAG)
- **Intelligent Question Answering**: Get direct answers to questions about your documents
- **Context-Aware Responses**: Responses are generated based on the content in your collection
- **Streaming Generation**: See answers appear in real-time as they're generated

### User-Friendly Interface
- **Streamlit Web Application**: Easy-to-use interface with intuitive controls
- **File Upload System**: Upload and index your own text and image content
- **Interactive Results**: Expandable text results and image galleries

## Architecture

The system consists of several key components:

1. **Content Processing Pipeline**
   - Text processing (TXT files)
   - Image processing (JPG, PNG files)
   - Metadata extraction

2. **Modality-Specific Encoders**
   - Text encoder: SentenceTransformers (all-MiniLM-L6-v2)
   - Image encoder: MobileNetV2

3. **Vector Database**
   - ChromaDB for efficient similarity search
   - Multi-modal indexing and retrieval

4. **Retrieval-Augmented Generation**
   - Google's Gemini API for response generation
   - Context formation from retrieved documents
   - Streaming response generation

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/multi-modal-retrieval-engine.git
cd multi-modal-retrieval-engine
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data/text data/images
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Using the application:
   - Click "Initialize Engine" to load the models
   - Upload text and/or image files and index them
   - Use the search tabs to find relevant content
   - Enter your Gemini API key in the RAG section
   - Ask questions about your content

### API Key Setup

To use the RAG functionality, you'll need a Google Generative AI API key:
1. Visit [Google AI Studio](https://ai.google.dev/)
2. Create an account and obtain an API key
3. Enter the key in the Gemini API Key Setup section of the application

## File Structure

```
multi-modal-retrieval-engine/
├── app.py                  # Streamlit web application
├── main.py                 # Command-line interface
├── rag_engine.py           # RAG functionality
├── requirements.txt        # Project dependencies
├── README.md               # This documentation
├── data/                   # Data directory
│   ├── text/               # Text documents
│   └── images/             # Image files
├── src/
│   ├── __init__.py
│   ├── text_processor.py   # Text embedding and processing
│   ├── image_processor.py  # Image embedding and processing
│   ├── retrieval_engine.py # Core retrieval functionality
│   └── rag_processor.py    # RAG integration with Gemini
```

## Current Limitations

1. **Cross-Modal Search Limitations**:
   - Text-to-image search uses a simplistic approach for bridging embedding spaces, resulting in less accurate results than a true cross-modal model would provide
   - No direct semantic understanding of image content beyond visual features

2. **Scalability Constraints**:
   - Limited to running on a local CPU
   - Performance degrades with large document collections
   - Embedding dimensionality (1280 for images, 384 for text) creates substantial memory requirements

3. **Content Type Restrictions**:
   - Only supports plain text files (.txt)
   - Limited image format support (JPG, PNG)
   - No handling of PDFs, Word documents, or structured data

4. **Language and Model Limitations**:
   - Text embeddings biased toward English language content
   - Lightweight models chosen for CPU compatibility sacrifice some accuracy
   - No entity recognition or knowledge graph capabilities

5. **Technical Constraints**:
   - Requires separate API key provisioning
   - No persistent storage beyond the local file system
   - In-memory database with no distributed options

## Future Enhancements

1. **Improved Multi-Modal Capabilities**:
   - Integration with CLIP or similar model for better cross-modal understanding
   - Image captioning to bridge visual and textual content
   - Support for audio and video content

2. **Extended Document Support**:
   - PDF processing with layout understanding
   - OCR for text extraction from images
   - Structured data (CSV, JSON, etc.) support

3. **Performance Optimizations**:
   - Model quantization for faster embedding generation
   - Batch processing for large document collections
   - Incremental indexing for new content

4. **Advanced Search Features**:
   - Hybrid search combining dense and sparse retrievers
   - Faceted search based on metadata
   - Semantic search with logical operators

5. **Enhanced User Experience**:
   - Custom knowledge base creation and management
   - User feedback loop to improve search quality
   - Session history and conversational search
   - Document annotation and collaborative features

6. **Infrastructure Improvements**:
   - Cloud deployment options
   - API endpoints for programmatic access
   - Authentication and multi-user support
   - Persistent vector database with backup capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [PyTorch](https://pytorch.org/) and [TorchVision](https://pytorch.org/vision/stable/index.html) for image processing
- [ChromaDB](https://www.trychroma.com/) for vector storage and retrieval
- [Streamlit](https://streamlit.io/) for the web interface
- [Google Generative AI](https://ai.google.dev/) for the RAG capabilities

---

*Note: This project was created for educational purposes and is not intended for production use without further development and security enhancements.*
