import os
from typing import List
from crewai_tools import BaseTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class PDFDirectoryLoaderTool(BaseTool):
    name: str = "PDF Directory Loader"
    description: str = "Loads all PDF files from a specified directory and returns their content"

    def _execute(self, directory_path: str) -> List[str]:
        pdf_contents = []
        
        # Walk through directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    try:
                        loader = PyPDFLoader(file_path)
                        pages = loader.load()
                        content = "\n".join([page.page_content for page in pages])
                        pdf_contents.append({
                            'filename': file,
                            'content': content
                        })
                    except Exception as e:
                        print(f"Error loading {file}: {str(e)}")
        
        return pdf_contents

class PDFAnalysisTool(BaseTool):
    name: str = "PDF Analysis Tool"
    description: str = "Analyzes PDF content using vector embeddings for semantic search and analysis"

    def __init__(self):
        super().__init__()
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def _execute(self, pdf_contents: List[dict], query: str = None) -> dict:
        all_chunks = []
        file_map = {}
        
        # Process each PDF
        for pdf in pdf_contents:
            chunks = self.text_splitter.split_text(pdf['content'])
            
            # Map chunks to filenames
            for chunk in chunks:
                all_chunks.append(chunk)
                file_map[chunk] = pdf['filename']
        
        # Create vector store
        vectorstore = FAISS.from_texts(all_chunks, self.embeddings)
        
        if query:
            # Perform semantic search
            results = vectorstore.similarity_search(query, k=5)
            return {
                'query_results': [
                    {
                        'content': doc.page_content,
                        'filename': file_map[doc.page_content]
                    } for doc in results
                ]
            }
        
        return {
            'chunks': all_chunks,
            'file_map': file_map
        } 