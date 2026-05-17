import os                                        # for file path and deletion
import tempfile                                  # for creating a safe temporary file on disk
from fastapi import APIRouter, UploadFile, File, HTTPException  # FastAPI building blocks
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # document parsers
from app.api.models import UploadResponse, ErrorResponse  # response shapes from Step 9
from app.services.vectorstore import VectorStore          # our vector store service from Step 5

router = APIRouter()

# create a single shared VectorStore instance for this module
# all upload requests use the same store so chunks accumulate correctly
_store = VectorStore()

# file extensions we support — anything else gets rejected
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


@router.post(
    "/documents/upload",                  # URL: POST /documents/upload
    response_model=UploadResponse,        # successful response shape
    tags=["Documents"],                   # groups under "Documents" in /docs
    summary="Upload a document to the knowledge base",
)
async def upload_document(
    file: UploadFile = File(...),         # File(...) means a file upload is required
):
    # extract the file extension and convert to lowercase (e.g. ".PDF" → ".pdf")
    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()

    # reject unsupported file types early with a clear HTTP 400 error
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {SUPPORTED_EXTENSIONS}",
        )

    # read the raw bytes of the uploaded file into memory
    contents = await file.read()          # 'await' because file reading is async in FastAPI

    # write the file to a temporary location on disk
    # delete=False means we manage deletion ourselves (needed on Windows)
    # suffix=ext preserves the extension so loaders know how to parse it
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)               # write the uploaded bytes to the temp file
        tmp_path = tmp.name               # save the path so we can use it after the block closes

    try:
        # choose the right loader based on file type
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)     # parses PDF pages into LangChain documents
        elif ext == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")  # parses plain text

        # load() returns a list of LangChain Document objects (one per page for PDFs)
        documents = loader.load()

        # tag each document with the original filename (not the temp path)
        # this is what shows up in citations: "source: DBMS_Notes.pdf"
        for doc in documents:
            doc.metadata["source"] = file.filename

        # count chunks before adding so we can report how many were indexed
        before_count = _store.index.ntotal if _store.index else 0

        # add the new documents to the live vector store
        _store.add_documents(documents)

        # count how many new chunks were added
        after_count = _store.index.ntotal
        chunks_added = after_count - before_count

        return UploadResponse(
            message=f"'{file.filename}' uploaded and indexed successfully.",
            chunks_added=chunks_added,
            filename=file.filename,
        )

    except Exception as e:
        # if anything goes wrong during loading or indexing, return a clean 500 error
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

    finally:
        # always delete the temp file, even if an error occurred above
        # 'finally' runs whether the try block succeeded or raised an exception
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
