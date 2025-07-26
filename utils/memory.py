# utils/memory.py

import streamlit as st
import json

def init_memory():
    """Initialize memory structure in session state."""
    if "memory" not in st.session_state:
        st.session_state.memory = {}

def remember(key: str, value):
    """Store a key-value pair in memory."""
    init_memory()
    st.session_state.memory[key] = value

def recall(key: str):
    """Retrieve a value from memory by key."""
    init_memory()
    return st.session_state.memory.get(key, None)

def forget(key: str):
    """Delete a specific memory key."""
    init_memory()
    if key in st.session_state.memory:
        del st.session_state.memory[key]

def clear_all_memory():
    """Clear the entire memory."""
    st.session_state.memory = {}

def show_memory():
    """Display memory contents in a JSON format."""
    init_memory()
    st.markdown("### 🧠 Current Memory")
    st.json(st.session_state.memory)
