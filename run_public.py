import os
import subprocess
import time
from pyngrok import ngrok

# 1. Open ngrok tunnel to port 8501
public_url = ngrok.connect(8501)
print("\n" + "=" * 60)
print(f"🌍 PUBLIC LINK FOR ANY DEVICE: {public_url}")
print("=" * 60 + "\n")

# 2. Launch Streamlit
os.system("streamlit run app.py")