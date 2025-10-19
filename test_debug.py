"""
Test and Debug Script for NWMSU Chatbot
Run: python test_debug.py
"""

import os
from neo4j import GraphDatabase
import warnings
warnings.filterwarnings("ignore")

# Configuration
NEO4J_URI = "neo4j+s://813403d3.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "4EfVPpL8RGgaSXTN1rudzLxMygGnihSAMtblyskNWz8"

class Neo4jDebugger:
    """Debug and test Neo4j connection and graph structure"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()
    
    def test_connection(self):
        """Test Neo4j connection"""
        print("🔌 Testing Neo4j connection...")
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record["test"] == 1:
                    print("✅ Connection successful!")
                    return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def get_node_counts(self):
        """Get count of all node types"""
        print("\n📊 Node Statistics:")
        print("-" * 50)
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """)
            
            total = 0
            for record in result:
                label = record["label"] or "No Label"
                count = record["count"]
                total += count
                print(f"  {label}: {count}")
            
            print(f"\n  Total Nodes: {total}")
            return total > 0
    
    def get_relationship_counts(self):
        """Get count of all relationship types"""
        print("\n🔗 Relationship Statistics:")
        print("-" * 50)
        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
            """)
            
            total = 0
            for record in result:
                rel_type = record["type"]
                count = record["count"]
                total += count
                print(f"  {rel_type}: {count}")
            
            print(f"\n  Total Relationships: {total}")
            return total > 0
    
    def show_sample_data(self):
        """Show sample nodes and relationships"""
        print("\n🔍 Sample Data:")
        print("-" * 50)
        
        with self.driver.session() as session:
            # Show Programs
            print("\n📚 Programs:")
            result = session.run("MATCH (p:Program) RETURN p LIMIT 5")
            for record in result:
                node = record["p"]
                print(f"  • {node.get('name', 'N/A')} (ID: {node.get('id', 'N/A')})")
            
            # Show Requirements
            print("\n📋 Requirements:")
            result = session.run("""
                MATCH (p:Program)-[:HAS_REQUIREMENT]->(r:Requirement)
                RETURN r.name as name, r.value as value, r.description as desc
            """)
            for record in result:
                print(f"  • {record['name']}: {record['value']}")
                if record['desc']:
                    print(f"    → {record['desc']}")
            
            # Show Faculty
            print("\n👨‍🏫 Faculty:")
            result = session.run("MATCH (f:Faculty) RETURN f")
            for record in result:
                node = record["f"]
                print(f"  • {node.get('name', 'N/A')}")
                print(f"    Role: {node.get('role', 'N/A')}")
                print(f"    Email: {node.get('email', 'N/A')}")
            
            # Show Documents
            print("\n📄 Document Chunks:")
            result = session.run("MATCH (d:Document) RETURN count(d) as count")
            count = result.single()["count"]
            print(f"  • Total document chunks: {count}")
    
    def test_vector_search(self):
        """Test vector similarity search"""
        print("\n🔎 Testing Vector Search:")
        print("-" * 50)
        
        with self.driver.session() as session:
            # Check if vector index exists
            result = session.run("""
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                WHERE type = 'VECTOR'
                RETURN name, labelsOrTypes, properties
            """)
            
            indexes = list(result)
            if indexes:
                print("✅ Vector indexes found:")
                for record in indexes:
                    print(f"  • {record['name']}: {record['labelsOrTypes']} - {record['properties']}")
            else:
                print("⚠️  No vector indexes found")
    
    def test_fulltext_search(self):
        """Test fulltext search"""
        print("\n🔍 Testing Fulltext Search:")
        print("-" * 50)
        
        with self.driver.session() as session:
            # Check if fulltext index exists
            result = session.run("""
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                WHERE type = 'FULLTEXT'
                RETURN name, labelsOrTypes, properties
            """)
            
            indexes = list(result)
            if indexes:
                print("✅ Fulltext indexes found:")
                for record in indexes:
                    print(f"  • {record['name']}: {record['labelsOrTypes']} - {record['properties']}")
                
                # Test a search
                try:
                    result = session.run("""
                        CALL db.index.fulltext.queryNodes('entity_search', 'Duolingo')
                        YIELD node, score
                        RETURN node.name as name, score
                        LIMIT 3
                    """)
                    print("\n  Test search for 'Duolingo':")
                    for record in result:
                        print(f"    → {record['name']} (score: {record['score']:.2f})")
                except Exception as e:
                    print(f"  ⚠️  Fulltext search test failed: {e}")
            else:
                print("⚠️  No fulltext indexes found")
    
    def show_graph_visualization_query(self):
        """Show Cypher query for visualization"""
        print("\n📈 Graph Visualization Query:")
        print("-" * 50)
        print("""
Copy this query into Neo4j Browser for visualization:

MATCH (p:Program)-[r]->(n)
RETURN p, r, n
UNION
MATCH (f:Faculty)-[r]->(p:Program)
RETURN f, r, p

Or for a simpler view:

MATCH (n)
WHERE n:Program OR n:Requirement OR n:Faculty
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 50
        """)
    
    def run_all_tests(self):
        """Run all diagnostic tests"""
        print("\n" + "=" * 60)
        print("🔬 NWMSU Chatbot Neo4j Diagnostics")
        print("=" * 60)
        
        # Test connection
        if not self.test_connection():
            print("\n❌ Cannot proceed - connection failed!")
            return False
        
        # Get statistics
        has_nodes = self.get_node_counts()
        has_relationships = self.get_relationship_counts()
        
        if not has_nodes:
            print("\n⚠️  WARNING: No nodes found in database!")
            print("   Run main.py to populate the database.")
            return False
        
        # Show sample data
        self.show_sample_data()
        
        # Test indexes
        self.test_vector_search()
        self.test_fulltext_search()
        
        # Show visualization query
        self.show_graph_visualization_query()
        
        print("\n" + "=" * 60)
        print("✅ Diagnostics Complete!")
        print("=" * 60)
        
        return True

def test_chatbot_qa():
    """Test the chatbot Q&A functionality"""
    print("\n" + "=" * 60)
    print("🤖 Testing Chatbot Q&A")
    print("=" * 60)
    
    try:
        from main import get_chatbot
        
        print("\nInitializing chatbot...")
        bot = get_chatbot()
        
        test_questions = [
            "What is the minimum Duolingo score?",
            "Who is the program advisor?",
            "What is the GPA requirement?",
            "What are the TOEFL requirements?"
        ]
        
        print("\nRunning test questions:\n")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Q: {question}")
            print("   Thinking...")
            
            try:
                answer = bot.ask(question)
                print(f"   A: {answer}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Q&A Tests Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Failed to test chatbot: {e}")
        print("   Make sure main.py is in the same directory")

def main():
    """Main test function"""
    print("""
╔════════════════════════════════════════════════════════════╗
║     NWMSU MS-ACS Chatbot - Test & Debug Tool              ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Run Neo4j diagnostics
    debugger = Neo4jDebugger()
    
    try:
        success = debugger.run_all_tests()
        
        if success:
            # Test chatbot
            response = input("\n\n🤔 Test chatbot Q&A functionality? (y/n): ")
            if response.lower() == 'y':
                test_chatbot_qa()
        
    except Exception as e:
        print(f"\n❌ Error during diagnostics: {e}")
    finally:
        debugger.close()
    
    print("\n\n📝 Next Steps:")
    print("  1. If graph is empty: Run 'python main.py' first")
    print("  2. If graph looks good: Run 'streamlit run app.py'")
    print("  3. For public access: Use 'npx localtunnel --port 8501'")
    print("\n")

if __name__ == "__main__":
    main()