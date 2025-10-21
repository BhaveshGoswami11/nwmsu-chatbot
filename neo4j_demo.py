#!/usr/bin/env python3
"""
Neo4j RAG Demo Script
Demonstrates the enhanced RAG system with Neo4j integration
"""

import os
import warnings
from typing import List, Tuple
import re

# LangChain imports
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

# Transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Neo4j
from neo4j import GraphDatabase

warnings.filterwarnings("ignore")

class Neo4jRAGChatbot:
    """Advanced RAG Chatbot with Neo4j Graph Database for NWMSU MS-ACS Program"""
    
    def __init__(self, use_gpu=False, use_neo4j=True):
        print("🚀 Initializing Neo4j RAG Chatbot...")
        
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.use_neo4j = use_neo4j
        print(f"📱 Device: {self.device}")
        print(f"🗄️ Neo4j Integration: {'Enabled' if use_neo4j else 'Disabled'}")
        
        # Initialize embeddings
        print("🔤 Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )
        
        # Initialize LLM
        print("🧠 Loading language model...")
        self.llm = self._setup_llm()
        
        # Initialize Neo4j connection
        if self.use_neo4j:
            self.graph = self._setup_neo4j()
        else:
            self.graph = None
        
        print("✅ Initialization complete!")
    
    def _setup_llm(self):
        """Setup the language model"""
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
    
    def _setup_neo4j(self):
        """Setup Neo4j connection"""
        print("🗄️ Setting up Neo4j connection...")
        
        # Neo4j connection details
        NEO4J_URI = "neo4j+s://813403d3.databases.neo4j.io"
        NEO4J_USERNAME = "neo4j"
        NEO4J_PASSWORD = "4EfVPpL8RGgaSXTN1rudzLxMygGnihSAMtblyskNWz8"
        
        try:
            # Test connection
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            with driver.session() as session:
                session.run("RETURN 1")
            
            # Create LangChain Neo4j graph
            graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD
            )
            
            print("✅ Neo4j connection established")
            return graph
            
        except Exception as e:
            print(f"⚠️ Neo4j connection failed: {e}")
            print("🔄 Falling back to FAISS-only mode")
            return None
    
    def load_data(self):
        """Load NWMSU MS-ACS data from web"""
        print("📥 Loading data from NWMSU website...")
        
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
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)
        
        print(f"✅ Loaded {len(chunks)} document chunks")
        return chunks
    
    def create_graph_structure(self, documents):
        """Create graph structure in Neo4j"""
        if not self.graph:
            print("⚠️ Neo4j not available, skipping graph creation")
            return
        
        print("🔧 Creating graph structure...")
        
        try:
            # Clear existing data
            self.graph.query("MATCH (n) DETACH DELETE n")
            
            # Create schema constraints
            self.graph.query("""
                CREATE CONSTRAINT program_id IF NOT EXISTS FOR (p:Program) REQUIRE p.id IS UNIQUE
            """)
            self.graph.query("""
                CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE
            """)
            self.graph.query("""
                CREATE CONSTRAINT course_id IF NOT EXISTS FOR (c:Course) REQUIRE c.id IS UNIQUE
            """)
            
            # Extract and create entities from documents
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
            
            print("✅ Graph structure created")
            
        except Exception as e:
            print(f"⚠️ Graph creation failed: {e}")
    
    def create_vector_store(self, documents):
        """Create vector store for semantic search"""
        print("🔍 Creating vector store...")
        
        if self.graph and self.use_neo4j:
            try:
                # Use Neo4j vector store
                vector_store = Neo4jVector.from_documents(
                    documents,
                    self.embeddings,
                    url="neo4j+s://813403d3.databases.neo4j.io",
                    username="neo4j",
                    password="4EfVPpL8RGgaSXTN1rudzLxMygGnihSAMtblyskNWz8",
                    index_name="msacs_documents",
                    node_label="Document",
                    text_node_property="text",
                    embedding_node_property="embedding"
                )
                print("✅ Neo4j vector store created")
                return vector_store
            except Exception as e:
                print(f"⚠️ Neo4j vector store failed: {e}")
                print("🔄 Falling back to FAISS")
        
        # Fallback to FAISS
        vector_store = FAISS.from_documents(documents, self.embeddings)
        print("✅ FAISS vector store created")
        return vector_store
    
    def retrieve_structured_context(self, question: str) -> str:
        """Retrieve structured data from Neo4j graph"""
        if not self.graph:
            return "Structured data not available (Neo4j not connected)"
        
        question_lower = question.lower()
        
        try:
            # Query for relevant entities based on question
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
    
    def retrieve_context(self, question: str, vector_store, k: int = 3) -> str:
        """Retrieve relevant context for the question"""
        print(f"🔍 Retrieving context for: '{question}'")
        
        # Get structured data from graph
        structured = self.retrieve_structured_context(question)
        
        # Get unstructured data from vector store
        docs = vector_store.similarity_search(question, k=k)
        unstructured = "\n\n".join([doc.page_content for doc in docs])
        
        # Combine both types of context
        context = f"Structured Information:\n{structured}\n\nAdditional Context:\n{unstructured}"
        
        print(f"📄 Retrieved {len(docs)} relevant documents")
        print(f"🗄️ Graph data: {'Available' if structured else 'Not available'}")
        return context
    
    def create_qa_chain(self, vector_store):
        """Create the question-answering chain"""
        print("⚙️ Creating QA chain...")
        
        def retriever_func(question: str) -> str:
            return self.retrieve_context(question, vector_store)
        
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
        
        print("✅ QA chain ready")
        return chain
    
    def setup(self):
        """Setup the complete RAG system"""
        print("🔧 Setting up Neo4j RAG system...")
        
        # Load data
        documents = self.load_data()
        
        # Create graph structure
        self.create_graph_structure(documents)
        
        # Create vector store
        self.vector_store = self.create_vector_store(documents)
        
        # Create QA chain
        self.qa_chain = self.create_qa_chain(self.vector_store)
        
        print("\n🎉 Neo4j RAG system ready!\n")
    
    def ask_with_context(self, question: str) -> Tuple[str, str]:
        """Ask a question and return both answer and context"""
        try:
            # Get context
            context = self.retrieve_context(question, self.vector_store)
            
            # Get answer
            answer = self.qa_chain.invoke({"question": question})
            
            return answer, context
        except Exception as e:
            return f"Error: {str(e)}", f"Context retrieval failed: {str(e)}"
    
    def get_graph_stats(self):
        """Get graph statistics"""
        if not self.graph:
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

def demo_neo4j_rag():
    """Demonstrate Neo4j RAG functionality"""
    print("🎓 NWMSU Neo4j RAG Demo")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = Neo4jRAGChatbot(use_gpu=False, use_neo4j=True)
    chatbot.setup()
    
    # Test questions
    questions = [
        "What is the minimum GPA required?",
        "What are the TOEFL requirements?",
        "Who is the program advisor?",
        "What is the Duolingo score requirement?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"❓ QUESTION: {question}")
        print(f"{'='*60}")
        
        answer, context = chatbot.ask_with_context(question)
        
        print(f"\n🔍 RETRIEVED CONTEXT:")
        print(f"{context[:500]}..." if len(context) > 500 else context)
        
        print(f"\n🤖 AI ANSWER:")
        print(f"{answer}")
    
    # Show graph stats
    print(f"\n📊 GRAPH STATISTICS:")
    stats = chatbot.get_graph_stats()
    for stat in stats:
        print(f"• {stat['label']}: {stat['count']}")

if __name__ == "__main__":
    demo_neo4j_rag()
