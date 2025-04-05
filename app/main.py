import streamlit as st
import requests
import json
from pathlib import Path
import time
import plotly.graph_objects as go

class RAGInterface:
    def __init__(self):
        self.API_URL = "http://localhost:8000"
        

    def _init_session_state(self):
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []


    def _display_documents(self):
        try:
            response = requests.get(f"{self.API_URL}/documents")
            if response.status_code == 200:
                docs = response.json()
                st.markdown("### 📄 Uploaded Documents")
                for doc in docs:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{doc['filename']}**: {doc['chunk_count']} chunks")
                    with col2:
                        if st.button("🗑️", key=f"delete-{doc['filename']}"):
                            try:
                                del_response = requests.delete(
                                    f"{self.API_URL}/documents/delete",
                                    json={"filename": doc['filename']}
                                )
                                if del_response.status_code == 200:
                                    st.success(f"Deleted! ")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Failed to delete document")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                st.warning("Failed to load document list")
        except Exception as e:
            st.error(f"Error fetching documents: {str(e)}")
    

    def _display_sources(self, sources):
        """Display sources in an expandable section."""
        with st.expander("View Sources", expanded=False):
            for idx, source in enumerate(sources, 1):
                st.markdown(f"**Source {idx}**")
                st.markdown(f"*From: {source['metadata']['source']}*")
                st.markdown(f"```\n{source['content']}\n```")
                st.markdown(f"Relevance Score: {source['rerank_score']:.2f}")
                st.markdown("---")

    def main(self):
        """Main Streamlit interface."""
        st.set_page_config(
            page_title="RAG Prototype",
            layout="wide"
        )
        
        self._init_session_state()
        
        # Title and description
        st.title("RAG System")
        st.markdown("""
        This system demonstrates a Retrieval-Augmented Generation (RAG) pipeline.
        Upload documents and ask questions about their content!
        """)
        
        # Sidebar
        with st.sidebar:
            st.header("📁 Document Upload")
            uploaded_file = st.file_uploader(
                "Upload a document",
                type=['pdf', 'txt', 'docx']
            )
            
            if uploaded_file:
                if st.button("📄 Process Document"):
                    with st.spinner("Processing document..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        try:
                            response = requests.post(
                                f"{self.API_URL}/upload",
                                files=files
                            )
                            ans = response.json()
                            if response.status_code == 200:
                                if ans["chunks"] == 0:
                                    st.info(f"⚠️ {ans['message']}")
                                else:
                                    st.success("Document processed successfully!")
                            else:
                                st.error("Error processing document")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            self._display_documents()

        # Main chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    self._display_sources(message["sources"])
        
        # Query input
        if prompt := st.chat_input("Ask about your documents"):
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        json_data = {"query": prompt, "top_k": 5}
                        response = requests.post(f"{self.API_URL}/query", json=json_data)
                    
                        if response.status_code == 200:
                            result = response.json()
                            st.markdown(result["answer"])
                            self._display_sources(result["sources"])
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"]
                            })
                        else:
                            st.error("Error getting response from the system")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    app = RAGInterface()
    app.main()