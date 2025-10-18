"""
Streamlit UI for NWMSU MS-ACS Chatbot
Save as: app.py
Run: streamlit run app.py
"""

import streamlit as st
from main import get_chatbot, ask_question, ask_question_with_context
import time

# Page configuration
st.set_page_config(
    page_title="NWMSU MS-ACS Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00563f;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #00563f 0%, #008556 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f1f8e9;
        border-left: 5px solid #4caf50;
    }
    
    .quick-question {
        background-color: #fff3e0;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 2px solid #ff9800;
        margin: 0.5rem;
        cursor: pointer;
        display: inline-block;
    }
    
    .stats-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #00563f;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot_loaded" not in st.session_state:
    st.session_state.chatbot_loaded = False

# Header
st.markdown('<div class="main-header">🎓 NWMSU MS-ACS Information Assistant</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.nwmissouri.edu/media/contentassets/img/wordmark.png", width=250)
    
    st.markdown("### About")
    st.info("""
    This chatbot provides information about Northwest Missouri State University's 
    **Master of Science in Applied Computer Science** program.
    
    Ask questions about:
    - 📋 Admission requirements
    - 📚 Courses and curriculum
    - 💰 Tuition and financial aid
    - 👨‍🏫 Faculty and advisors
    - 🏢 Campus facilities
    """)
    
    st.markdown("---")
    
    # Quick questions
    st.markdown("### 💡 Quick Questions")
    quick_questions = [
        "What is the minimum GPA required?",
        "What is the Duolingo score requirement?",
        "Who is the program advisor?",
        "What are the TOEFL requirements?",
        "How can I contact the department?",
        "What courses are offered?"
    ]
    
    for q in quick_questions:
        if st.button(q, key=f"quick_{q}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()
    
    st.markdown("---")
    
    # Graph stats
    if st.button("📊 Show Graph Stats"):
        try:
            bot = get_chatbot()
            stats = bot.get_graph_stats()
            st.markdown("### Graph Statistics")
            for stat in stats:
                st.write(f"**{stat['label']}**: {stat['count']}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
st.markdown("### 💬 Chat")

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <b>🧑 You:</b><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <b>🤖 Assistant:</b><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Chat input (outside columns)
if prompt := st.chat_input("Ask me anything about MS-ACS program..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message immediately
    st.markdown(f"""
    <div class="chat-message user-message">
        <b>🧑 You:</b><br>
        {prompt}
    </div>
    """, unsafe_allow_html=True)
    
    # Show loading
    with st.spinner("🤔 Thinking..."):
        try:
            # Initialize chatbot if needed
            if not st.session_state.chatbot_loaded:
                with st.status("Initializing chatbot...", expanded=True) as status:
                    st.write("Loading models...")
                    bot = get_chatbot()
                    st.session_state.chatbot_loaded = True
                    status.update(label="✅ Ready!", state="complete")
            
            # Get response with context
            response, context = ask_question_with_context(prompt)
            
            # Show retrieved context
            with st.expander("🔍 Retrieved Context (RAG)", expanded=False):
                st.markdown("**This is the context retrieved from the knowledge base before generating the answer:**")
                st.text_area("Context:", context, height=200, disabled=True)
            
            # Add bot response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to show new message
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Sorry, I encountered an error: {str(e)}"
            })
            
            # Show error context
            with st.expander("🔍 Error Details", expanded=False):
                st.code(str(e), language="text")

# Main layout with columns
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📌 Tips")
    st.markdown("""
    <div class="stats-box">
    <b>Get better answers:</b>
    <ul>
        <li>Be specific in your questions</li>
        <li>Ask one question at a time</li>
        <li>Use keywords like "requirement", "admission", "course"</li>
    </ul>
    
    <b>Example questions:</b>
    <ul>
        <li>"What are the English proficiency requirements?"</li>
        <li>"Tell me about the admission process"</li>
        <li>"What is the program duration?"</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Status indicator
    if st.session_state.chatbot_loaded:
        st.success("✅ Chatbot Ready")
    else:
        st.warning("⏳ Chatbot will load on first question")
    
    st.metric("Messages", len(st.session_state.messages))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>
    🎓 Northwest Missouri State University<br>
    Master of Science in Applied Computer Science<br>
    <a href="https://www.nwmissouri.edu/csis/msacs/" target="_blank">Official Program Website</a>
    </small>
</div>
""", unsafe_allow_html=True)