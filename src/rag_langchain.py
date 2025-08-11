import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
  from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
  from langchain_community.embeddings import HuggingFaceEmbeddings

try:
  from langchain_chroma import Chroma
except ImportError:
  from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from multiprocessing.sharedctypes import Synchronized as SynchronizedClass

class RAGLangchain:
  def __init__(self, interrupt_count: SynchronizedClass):
    self.logger = logging.getLogger("speech_to_speech.rag_langchain")
    self.interrupt_count = interrupt_count

    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    self.db = Chroma(
      collection_name="RAG",
      embedding_function=self.embeddings
      )
    self.splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100, length_function=len,  is_separator_regex=False)


  def add_document(self, document):
    # Validate document content
    if not document or "content" not in document:
      self.logger.warning("Document is missing or has no 'content' field")
      return
    
    content = document["content"]
    if not content or not content.strip():
      self.logger.warning("Document content is empty or whitespace only")
      return
    
    # Validate source field
    source = document.get("source", "unknown")
    
    documents = []
    chunks = self.splitter.split_text(content)
    
    for chunk in chunks:
      # Only add non-empty chunks
      if chunk and chunk.strip():
        documents.append(Document(chunk.strip(), metadata={"source": source}))
    
    if documents:
      try:
        self.db.add_documents(documents)
        self.logger.debug(f"Added {len(documents)} chunks from source: {source}")
      except Exception as e:
        self.logger.error(f"Failed to add documents to RAG: {e}")
    else:
      self.logger.warning(f"No valid chunks generated from document: {source}")


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
  