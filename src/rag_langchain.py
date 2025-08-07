import logging
import warnings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

logger = logging.getLogger("speech_to_speech.rag_langchain")

class RAGLangchain:
  def __init__(self):
    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    self.db = Chroma(
      collection_name="RAG",
      embedding_function=self.embeddings
      )
    self.splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100, length_function=len,  is_separator_regex=False)

  def add_document(self, document):
    # Validate document content
    if not document or not document.get("content") or not document["content"].strip():
      logger.warning(f"Skipping empty document from {document.get('source', 'unknown')}")
      return
    
    documents = []
    chunks = self.splitter.split_text(document["content"])
    
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk.strip()]
    
    if not valid_chunks:
      logger.warning(f"No valid chunks found in document from {document.get('source', 'unknown')}")
      return
    
    for chunk in valid_chunks:
      documents.append(Document(chunk, metadata={"source": document["source"]}))
    
    logger.debug(f"Adding {len(documents)} document chunks to RAG")
    self.db.add_documents(documents)

  def query(self, prompt):
    results = []
    
    search_results = self.db.similarity_search_with_relevance_scores(
      query=prompt,
      k = 5
    )
    
    for document, score in search_results:
      results.append({
        "content": document.page_content,
        "source": document.metadata["source"],
        "score": score
      })

    return results
  