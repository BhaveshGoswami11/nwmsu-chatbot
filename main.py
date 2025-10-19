"""
Improved RAG System with Streamlit UI and Proper Graph Construction
Run: streamlit run app.py
Then use: npx localtunnel --port 8501
"""

# ============= main.py =============
import os
import warnings
from typing import List, Tuple
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.chains import RetrievalQA  # Not used in current implementation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re

warnings.filterwarnings("ignore")

# Neo4j Configuration
os.environ["NEO4J_URI"] = "neo4j+s://813403d3.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "4EfVPpL8RGgaSXTN1rudzLxMygGnihSAMtblyskNWz8"

class ImprovedRAGChatbot:
    """Improved RAG chatbot with proper graph construction"""
    
    def __init__(self, use_gpu=False):
        print("🚀 Initializing Chatbot...")
        
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        # Initialize embeddings
        print("Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )
        
        # Initialize LLM
        print("Loading LLM...")
        self.llm = self._setup_llm()
        
        # Connect to Neo4j
        try:
            self.graph = Neo4jGraph(
                url=os.environ["NEO4J_URI"],
                username=os.environ["NEO4J_USERNAME"],
                password=os.environ["NEO4J_PASSWORD"]
            )
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            self.graph = None
        
        print("✓ Initialization complete!")
    
    def _setup_llm(self):
        """Setup LLM"""
        model_id = "google/flan-t5-base"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            device=0 if self.device == "cuda" else -1
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def load_data(self):
        """Load NWMSU data"""
        print("📥 Loading data...")
        
        urls = [
            "https://www.nwmissouri.edu/csis/msacs/",
            "https://www.nwmissouri.edu/csis/msacs/about.htm",
            "https://www.nwmissouri.edu/academics/graduate/masters/applied-computer-science.htm",
            "https://www.nwmissouri.edu/csis/msacs/apply/index.htm",
            "https://www.nwmissouri.edu/csis/msacs/courses.htm",
            "https://www.nwmissouri.edu/csis/msacs/FAQs.htm",
            "https://www.nwmissouri.edu/csis/msacs/contact.htm"
        ]
        
        loader = WebBaseLoader(urls)
        loader.requests_kwargs = {'verify': False}
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)
        
        print(f"✓ Loaded {len(chunks)} chunks")
        return chunks
    
    def create_graph_structure(self, documents):
        """Create proper graph structure in Neo4j"""
        print("🔧 Creating graph structure...")
        
        if self.graph is None:
            print("Warning: Neo4j not available, skipping graph creation")
            return
        
        # Clear existing data
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
        except Exception as e:
            print(f"Warning: Could not clear Neo4j data: {e}")
            return
        
        # Create schema
        self.graph.query("""
            CREATE CONSTRAINT program_id IF NOT EXISTS FOR (p:Program) REQUIRE p.id IS UNIQUE
        """)
        self.graph.query("""
            CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE
        """)
        self.graph.query("""
            CREATE CONSTRAINT course_id IF NOT EXISTS FOR (c:Course) REQUIRE c.id IS UNIQUE
        """)
        
        # Extract and create entities
        for doc in documents:
            content = doc.page_content.lower()
            
            # Create Program node
            if "ms-acs" in content or "applied computer science" in content:
                self.graph.query("""
                    MERGE (p:Program {id: 'MS-ACS'})
                    SET p.name = 'Master of Science in Applied Computer Science',
                        p.description = 'Graduate program in Applied Computer Science'
                """)
            
            # Create requirement nodes
            if "gpa" in content and "2.75" in content:
                self.graph.query("""
                    MERGE (r:Requirement {id: 'GPA'})
                    SET r.name = 'Minimum GPA',
                        r.value = '2.75',
                        r.description = 'Overall undergraduate GPA requirement'
                    WITH r
                    MATCH (p:Program {id: 'MS-ACS'})
                    MERGE (p)-[:HAS_REQUIREMENT]->(r)
                """)
            
            if "duolingo" in content:
                match = re.search(r'duolingo[^\d]*(\d+)', content)
                if match:
                    score = match.group(1)
                    self.graph.query("""
                        MERGE (r:Requirement {id: 'Duolingo'})
                        SET r.name = 'Duolingo English Test',
                            r.value = $score,
                            r.description = 'Minimum Duolingo score requirement'
                        WITH r
                        MATCH (p:Program {id: 'MS-ACS'})
                        MERGE (p)-[:HAS_REQUIREMENT]->(r)
                    """, {'score': score})
            
            if "toefl" in content:
                self.graph.query("""
                    MERGE (r:Requirement {id: 'TOEFL'})
                    SET r.name = 'TOEFL',
                        r.ibt = '71',
                        r.pbt = '550',
                        r.cbt = '213',
                        r.description = 'TOEFL score requirements'
                    WITH r
                    MATCH (p:Program {id: 'MS-ACS'})
                    MERGE (p)-[:HAS_REQUIREMENT]->(r)
                """)
            
            if "ielts" in content:
                self.graph.query("""
                    MERGE (r:Requirement {id: 'IELTS'})
                    SET r.name = 'IELTS',
                        r.value = '6.0',
                        r.description = 'Minimum IELTS score'
                    WITH r
                    MATCH (p:Program {id: 'MS-ACS'})
                    MERGE (p)-[:HAS_REQUIREMENT]->(r)
                """)
            
            # Create advisor node
            if "ajay" in content or "bandi" in content:
                self.graph.query("""
                    MERGE (a:Faculty {id: 'Dr. Ajay Bandi'})
                    SET a.name = 'Dr. Ajay Bandi',
                        a.role = 'Program Advisor',
                        a.email = 'ajay@nwmissouri.edu'
                    WITH a
                    MATCH (p:Program {id: 'MS-ACS'})
                    MERGE (a)-[:ADVISES]->(p)
                """)
        
        # Create fulltext index for searching
        self.graph.query("""
            CREATE FULLTEXT INDEX entity_search IF NOT EXISTS 
            FOR (n:Program|Requirement|Faculty|Course) 
            ON EACH [n.id, n.name, n.description]
        """)
        
        print("✓ Graph structure created")
    
    def setup_vector_store(self, documents):
        """Create vector store"""
        print("🔧 Setting up vector store...")
        
        try:
            vector_store = Neo4jVector.from_documents(
                documents,
                self.embeddings,
                url=os.environ["NEO4J_URI"],
                username=os.environ["NEO4J_USERNAME"],
                password=os.environ["NEO4J_PASSWORD"],
                index_name="msacs_documents",
                node_label="Document",
                text_node_property="text",
                embedding_node_property="embedding"
            )
            print("✓ Vector store ready")
            return vector_store
        except Exception as e:
            print(f"Warning: Could not create Neo4j vector store: {e}")
            print("Falling back to in-memory vector store...")
            # Fallback to a simple in-memory vector store
            from langchain_community.vectorstores import FAISS
            vector_store = FAISS.from_documents(documents, self.embeddings)
            print("✓ Fallback vector store ready")
            return vector_store
    
    def structured_retriever(self, question: str) -> str:
        """Retrieve structured data from graph"""
        if self.graph is None:
            return "Structured data not available (Neo4j not connected)"
            
        question_lower = question.lower()
        
        try:
            # Query for relevant entities
            if "duolingo" in question_lower:
                result = self.graph.query("""
                    MATCH (p:Program)-[:HAS_REQUIREMENT]->(r:Requirement {id: 'Duolingo'})
                    RETURN r.name + ': ' + r.value + ' - ' + r.description AS info
                """)
            elif "toefl" in question_lower:
                result = self.graph.query("""
                    MATCH (p:Program)-[:HAS_REQUIREMENT]->(r:Requirement {id: 'TOEFL'})
                    RETURN r.name + ' IBT: ' + r.ibt + ', PBT: ' + r.pbt AS info
                """)
            elif "gpa" in question_lower:
                result = self.graph.query("""
                    MATCH (p:Program)-[:HAS_REQUIREMENT]->(r:Requirement {id: 'GPA'})
                    RETURN r.name + ': ' + r.value AS info
                """)
            elif "advisor" in question_lower or "contact" in question_lower:
                result = self.graph.query("""
                    MATCH (a:Faculty)-[:ADVISES]->(p:Program {id: 'MS-ACS'})
                    RETURN a.name + ' (' + a.role + '): ' + a.email AS info
                """)
            else:
                result = self.graph.query("""
                    MATCH (p:Program {id: 'MS-ACS'})-[r]->(n)
                    RETURN type(r) + ': ' + coalesce(n.name, n.id) AS info
                    LIMIT 5
                """)
            
            return "\n".join([r['info'] for r in result]) if result else ""
        except Exception as e:
            return f"Error retrieving structured data: {str(e)}"
    
    def create_qa_chain(self, vector_store):
        """Create QA chain"""
        print("⚙️ Creating QA chain...")
        
        def retriever_func(question: str) -> str:
            # Get structured data from graph
            structured = self.structured_retriever(question)
            
            # Get unstructured data from vector store
            docs = vector_store.similarity_search(question, k=3)
            unstructured = "\n\n".join([d.page_content for d in docs])
            
            return f"Structured Information:\n{structured}\n\nAdditional Context:\n{unstructured}"
        
        template = """Answer the question based on the context below. Be specific and accurate.

Context: {context}

Question: {question}

Answer concisely:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            RunnableParallel({
                "context": lambda x: retriever_func(x["question"]),
                "question": lambda x: x["question"]
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ QA chain ready")
        return chain
    
    def setup(self):
        """Setup complete system"""
        documents = self.load_data()
        self.create_graph_structure(documents)
        self.vector_store = self.setup_vector_store(documents)
        self.qa_chain = self.create_qa_chain(self.vector_store)
        print("\n✅ System ready!\n")
    
    def ask(self, question: str) -> str:
        """Ask a question"""
        try:
            return self.qa_chain.invoke({"question": question})
        except Exception as e:
            return f"Error: {str(e)}"
    
    def ask_with_context(self, question: str) -> tuple[str, str]:
        """Ask a question and return both answer and context"""
        try:
            # Get structured data from graph
            structured = self.structured_retriever(question)
            
            # Get unstructured data from vector store
            docs = self.vector_store.similarity_search(question, k=3)
            unstructured = "\n\n".join([d.page_content for d in docs])
            
            # Combine context
            context = f"Structured Information:\n{structured}\n\nAdditional Context:\n{unstructured}"
            
            # Get answer
            answer = self.qa_chain.invoke({"question": question})
            
            return answer, context
        except Exception as e:
            return f"Error: {str(e)}", f"Context retrieval failed: {str(e)}"
    
    def get_graph_stats(self):
        """Get graph statistics"""
        if self.graph is None:
            return [{"label": "Neo4j", "count": "Not connected"}]
        
        try:
            stats = self.graph.query("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """)
            return stats
        except Exception as e:
            return [{"label": "Error", "count": str(e)}]

# Global chatbot instance
_chatbot = None

def get_chatbot():
    """Get or create chatbot instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = ImprovedRAGChatbot(use_gpu=False)
        _chatbot.setup()
    return _chatbot

def ask_question(question: str) -> str:
    """Public API for asking questions"""
    bot = get_chatbot()
    return bot.ask(question)

def ask_question_with_context(question: str) -> tuple[str, str]:
    """Public API for asking questions with context"""
    bot = get_chatbot()
    return bot.ask_with_context(question)

if __name__ == "__main__":
    bot = get_chatbot()
    
    # Test questions
    test_qs = [
        "What is the minimum Duolingo score?",
        "Who is the program advisor?",
        "What is the GPA requirement?"
    ]
    
    for q in test_qs:
        print(f"\nQ: {q}")
        print(f"A: {bot.ask(q)}")