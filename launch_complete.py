"""
Launcher for Complete Video Classifier App
"""

import subprocess
import sys
import time
import webbrowser

print("🚀 Launching Complete Video Classifier...")
print("⏳ Starting Streamlit server...")

# Start the Streamlit server
process = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run",
    "final_complete_app.py",
    "--server.port", "8501",
    "--logger.level=error"
])

# Wait for server to start
time.sleep(3)

# Open browser
print("🌐 Opening browser... http://localhost:8501")
try:
    webbrowser.open("http://localhost:8501")
except:
    print("Please open http://localhost:8501 in your browser manually")

print("\n✅ App is running!")
print("Press Ctrl+C to stop the server\n")

# Keep the process running
try:
    process.wait()
except KeyboardInterrupt:
    print("\n🛑 Stopping server...")
    process.terminate()
    process.wait()