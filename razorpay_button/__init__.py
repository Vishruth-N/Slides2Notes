import streamlit.components.v1 as components
import os

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def razorpay_button():
    components.html(
        open(os.path.join(_CURRENT_DIR, "razorpay.html")).read(),
        height=100,
    )
