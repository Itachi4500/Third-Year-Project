import streamlit as st

def select_dashboard_size():
    size_option = st.selectbox("Select Dashboard Size", ["Default (1280x720)", "4:3 (960x720)", "Letter (816x1056)", "Custom"])
    
    if size_option == "Default (1280x720)":
        return (1280, 720)
    elif size_option == "4:3 (960x720)":
        return (960, 720)
    elif size_option == "Letter (816x1056)":
        return (816, 1056)
    elif size_option == "Custom":
        width = st.number_input("Width", min_value=500, value=1280)
        height = st.number_input("Height", min_value=500, value=720)
        return (width, height)
