import subprocess

def run_api():
    # Command to run the FastAPI application
    subprocess.Popen(["uvicorn", "src.api.api:app", "--reload", "--host", "0.0.0.0", "--port", "8001"])

def run_gui():
    # Command to run the GUI application using Streamlit
    subprocess.Popen(["streamlit", "run", "src/gui/gui.py"])

if __name__ == "__main__":
    run_api()
    run_gui()

    # Keeping the script running to allow both processes to stay active
    print("API is running on http://localhost:8001")
    print("GUI is running on http://localhost:8501")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping services...")
