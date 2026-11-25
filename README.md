# PDF RAG Application

A **Retrieval-Augmented Generation (RAG)** application built using FastAPI, LangChain, Qdrant, and OpenRouter. It allows users to upload PDF documents and ask questions about their content using AI.

## Features

- **PDF Upload:** Upload PDF documents through FastAPI.
- **PDF Processing:** Extract text and split documents into smaller chunks.
- **Text Embeddings:** Generate embeddings using Cohere Embed 5 Fast through OpenRouter.
- **Vector Database:** Store and retrieve document embeddings using Qdrant.
- **AI Question Answering:** Ask questions about uploaded PDFs using Llama through OpenRouter.
- **Context-Based Responses:** Generate answers using relevant information retrieved from the PDF.
- **Source References:** Return relevant document content along with page numbers.
- **Combined Upload and Query:** Upload a PDF and ask a question in a single API request.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| FastAPI | REST API development |
| LangChain | Document loading and text splitting |
| PyPDF | PDF text extraction |
| Cohere Embed 5 Fast | Text embeddings |
| OpenRouter | Access to embedding and LLM APIs |
| Llama 3.3 | AI answer generation |
| Qdrant | Vector database |
| Docker | Running Qdrant locally |

## Project Structure

```text
rag_project/
│
├── app/
│   ├── routers/
│   │   ├── pdf.py
│   │   └── query.py
│   │
│   ├── uploads/
│   │
│   ├── embeddings.py
│   └── main.py
│
├── .env
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd rag_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install fastapi uvicorn openai python-dotenv python-multipart pypdf langchain-community langchain-core langchain-text-splitters langchain-qdrant
```

### 4. Configure OpenRouter API Key

Create a `.env` file in the project root.

```env
OPENROUTER_KEY=your_openrouter_api_key
```

Get your API key from:

https://openrouter.ai/settings/keys

The application uses:
- `cohere/embed-5-fast` for generating embeddings.
- `meta-llama/llama-3.3-70b-instruct` for generating answers.

Check model availability and pricing on OpenRouter before using them.

### 5. Start Qdrant

Make sure Docker Desktop is running.

Start your Qdrant container:

```bash
docker compose up -d
```

Check whether the container is running:

```bash
docker ps
```

Qdrant dashboard:

http://localhost:6333/dashboard

### 6. Run the FastAPI application

From the project root:

```bash
python -m uvicorn app.main:app --reload
```

The application will run at:

http://127.0.0.1:8000

## API Endpoints

### 1. Upload a PDF

**Endpoint:**

```http
POST /pdf/upload
```

Uploads a PDF, extracts its content, generates embeddings, and stores them in a dedicated Qdrant collection.

**Input:**
- PDF file

**Response:**

```json
{
  "message": "PDF uploaded and processed successfully.",
  "file_id": "generated-file-id",
  "filename": "sample.pdf",
  "total_pages": 10,
  "total_chunks": 18,
  "collection_name": "pdf_generatedid"
}
```

### 2. Query an Uploaded PDF

**Endpoint:**

```http
POST /query/{collection_name}?question=Your question
```

Retrieves relevant chunks from the selected PDF collection and generates an AI response.

**Example:**

```http
POST /query/pdf_generatedid?question=What is this PDF about?
```

### 3. Upload and Query in One Request

**Endpoint:**

```http
POST /pdf/upload-and-query
```

Allows users to upload a PDF and ask a question in the same request.

**Input:**
- `file`: PDF document
- `question`: Question about the document

**Example question:**

```text
Explain the main concepts discussed in this document.
```

**Response:**

```json
{
  "filename": "sample.pdf",
  "question": "What is this PDF about?",
  "answer": "The PDF discusses...",
  "sources": [
    {
      "page": 1,
      "content": "Relevant extracted PDF content..."
    }
  ]
}
```

## How the Application Works

The application follows a Retrieval-Augmented Generation workflow.

1. **PDF Upload:** The user uploads a PDF through a FastAPI endpoint.
2. **Text Extraction:** PyPDFLoader extracts text and page metadata.
3. **Text Chunking:** RecursiveCharacterTextSplitter divides the extracted text into smaller chunks.
4. **Embedding Generation:** Cohere Embed 5 Fast converts the text chunks into numerical vectors through OpenRouter.
5. **Vector Storage:** Qdrant stores the generated vectors and associated document content.
6. **Similarity Search:** When a user asks a question, Qdrant retrieves the most relevant document chunks.
7. **Answer Generation:** The retrieved context and user's question are sent to the Llama model through OpenRouter.
8. **Final Response:** The API returns an AI-generated answer along with relevant source content and page numbers.

## Testing the API

FastAPI provides an interactive Swagger UI for testing endpoints.

Open:

http://127.0.0.1:8000/docs

You can:
- Upload PDF documents.
- Submit questions about uploaded PDFs.
- Test the combined upload-and-query endpoint.
- View API responses and source references.

## Important Notes

- Qdrant must be running before using the API.
- An OpenRouter API key is required.
- API usage may incur charges depending on the selected models and available free credits.
- The application currently supports text-based PDFs. Scanned PDFs require OCR processing.
- Each separately uploaded PDF gets its own Qdrant collection.
- The combined upload-and-query endpoint processes a new PDF for each request. For multiple questions about the same document, use the separate upload and query endpoints.
- Do not upload your `.env` file or API keys to GitHub.

## Future Improvements

- Support multiple PDF uploads.
- Add OCR support for scanned documents.
- Build a frontend chatbot interface.
- Add conversation history.
- Implement PDF deletion and collection management.
- Add streaming AI responses.
- Support multiple users and document access control.

---

**Built with Python, FastAPI, LangChain, Qdrant, and OpenRouter.**
