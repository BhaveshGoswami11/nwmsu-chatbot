# NWMSU MS-ACS RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that provides information about Northwest Missouri State University's Master of Science in Applied Computer Science program. This chatbot demonstrates how to provide context to AI models before answering questions.

## Features

- **RAG Implementation**: Shows retrieved context before generating answers
- **Graph Database**: Uses Neo4j for structured data storage
- **Vector Search**: Uses embeddings for semantic document retrieval
- **Streamlit UI**: Modern web interface with context visualization
- **Fallback Support**: Works even without Neo4j connection

## Architecture

```
User Question → Context Retrieval → LLM Generation → Answer
     ↓              ↓                    ↓
  Vector Store   Graph Database      Response
  (Documents)    (Structured Data)   (With Context)
```

## Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- Internet connection for downloading models

## Installation

1. **Clone or download the project**
   ```bash
   cd /Users/Downloads/projects/nwmsu-chatbot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (optional)**
   ```bash
   export NEO4J_URI=""
   export NEO4J_USERNAME="neo4j"
   export NEO4J_PASSWORD=""
   ```

## Running the Application

### Method 1: Direct Streamlit Run
```bash
streamlit run app.py
```

### Method 2: With Port Specification
```bash
streamlit run app.py --server.port 8501
```

### Method 3: With Public URL (using localtunnel)
```bash
# Terminal 1: Run the app
streamlit run app.py

# Terminal 2: Create public URL
npx localtunnel --port 8501
```

## Usage

1. **Open the application** in your browser (usually `http://localhost:8501`)

2. **Ask questions** about the MS-ACS program:
   - "What is the minimum GPA required?"
   - "What are the TOEFL requirements?"
   - "Who is the program advisor?"
   - "What courses are offered?"

3. **View RAG context**: Click on "🔍 Retrieved Context (RAG)" to see the context retrieved before generating the answer

4. **Use quick questions**: Click on the predefined questions in the sidebar

## RAG Demonstration

The chatbot demonstrates RAG by:

1. **Retrieving Context**: Shows both structured data (from Neo4j graph) and unstructured data (from vector store)
2. **Context Display**: Users can see exactly what information was retrieved before the answer was generated
3. **Transparency**: The retrieval process is visible, showing how the AI uses context to answer questions

## Project Structure

```
nwmsu-chatbot/
├── app.py              # Streamlit UI
├── main.py             # RAG implementation
├── requirements.txt    # Dependencies
├── README.md          # This file
└── venv/              # Virtual environment
```

## Key Components

### 1. Data Loading (`main.py`)
- Loads NWMSU MS-ACS web pages
- Splits documents into chunks
- Creates embeddings

### 2. Graph Construction (`main.py`)
- Creates Neo4j graph with program information
- Establishes relationships between entities
- Handles connection failures gracefully

### 3. Vector Store (`main.py`)
- Stores document embeddings
- Enables semantic search
- Falls back to FAISS if Neo4j unavailable

### 4. RAG Chain (`main.py`)
- Combines structured and unstructured retrieval
- Uses retrieved context for answer generation
- Provides context transparency

### 5. UI (`app.py`)
- Modern Streamlit interface
- Context visualization
- Error handling and user feedback

## Troubleshooting

### Common Issues

1. **Model Download Takes Time**
   - First run downloads models (~1GB)
   - Subsequent runs are faster

2. **Neo4j Connection Issues**
   - App works without Neo4j (uses fallback)
   - Check internet connection

3. **Memory Issues**
   - Models require ~2GB RAM
   - Close other applications if needed

4. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

### Error Messages

- **"Could not connect to Neo4j"**: Normal if Neo4j unavailable
- **"Model loading..."**: First-time setup, be patient
- **"Context retrieval failed"**: Check internet connection

## Customization

### Adding New Data Sources
1. Update URLs in `load_data()` method
2. Modify graph structure in `create_graph_structure()`
3. Adjust retrieval logic in `structured_retriever()`

### Changing the LLM
1. Update model in `_setup_llm()` method
2. Adjust prompt template in `create_qa_chain()`
3. Update requirements.txt if needed

### UI Modifications
1. Edit CSS in `app.py`
2. Add new quick questions
3. Modify layout and styling

## Technical Details

- **LLM**: Google Flan-T5-Base
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: Neo4j Vector (with FAISS fallback)
- **Graph Database**: Neo4j
- **UI Framework**: Streamlit
- **Python Version**: 3.9+

## Performance

- **Initial Load**: ~30-60 seconds (model download)
- **Question Response**: ~2-5 seconds
- **Memory Usage**: ~2GB RAM
- **Storage**: ~1GB for models

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages in the UI
3. Ensure all dependencies are installed
4. Verify internet connection for model downloads

## License

This project is for educational purposes demonstrating RAG implementation.
