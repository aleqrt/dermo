"""
Simple Flask API for skin lesion classification model training and testing.
Uses subprocesses to run train.py and test.py scripts.
"""
import os
import sys
import uuid
import shlex
import threading
import subprocess
import time
from pathlib import Path
from flask import Flask, request, jsonify

# Setup paths
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
SRC_DIR = ROOT_DIR / "src"
PYTHON_EXE = sys.executable

# Add src directory to Python path if needed
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Initialize Flask app
app = Flask(__name__)

# In-memory job storage
jobs = {}

def run_job(command, job_id):
    """Run a command as a subprocess and capture output."""
    jobs[job_id]["status"] = "running"
    jobs[job_id]["start_time"] = time.time()
    
    process = subprocess.Popen(
        shlex.split(command),
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Capture logs
    log_lines = []
    for line in iter(process.stdout.readline, ""):
        log_lines.append(line)
        jobs[job_id]["logs"] = "".join(log_lines)
    
    # Process complete
    return_code = process.wait()
    jobs[job_id]["status"] = "completed" if return_code == 0 else "failed"
    jobs[job_id]["return_code"] = return_code
    jobs[job_id]["end_time"] = time.time()

@app.route("/train", methods=["POST"])
def start_training():
    """Start a training job with the parameters from the request."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "type": "training",
        "status": "initializing",
        #"logs": "",
        "created_at": time.time()
    }
    
    # Get training parameters from request
    params = request.get_json(silent=True) or {}
    
    # Build command with parameters
    cmd = [PYTHON_EXE, f"{SRC_DIR}/train.py"]
    for key, value in params.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    command = " ".join(shlex.quote(str(arg)) for arg in cmd)
    jobs[job_id]["command"] = command
    
    # Start job in a background thread
    thread = threading.Thread(
        target=run_job, 
        args=(command, job_id), 
        daemon=True
    )
    thread.start()
    
    return jsonify({
        "job_id": job_id,
        "status": "started",
        "type": "training",
        "message": "Training job started"
    }), 202

@app.route("/test", methods=["POST"])
def start_testing():
    """Start a testing job with the parameters from the request."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "type": "testing",
        "status": "initializing",
        #"logs": "",
        "created_at": time.time()
    }
    
    # Get testing parameters from request
    params = request.get_json(silent=True) or {}
    
    # Build command with parameters
    cmd = [PYTHON_EXE, f"{SRC_DIR}/test.py"]
    for key, value in params.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    command = " ".join(shlex.quote(str(arg)) for arg in cmd)
    jobs[job_id]["command"] = command
    
    # Start job in a background thread
    thread = threading.Thread(
        target=run_job, 
        args=(command, job_id), 
        daemon=True
    )
    thread.start()
    
    return jsonify({
        "job_id": job_id,
        "status": "started",
        "type": "testing",
        "message": "Testing job started"
    }), 202

@app.route("/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Get the status of a job."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    response = {
        "job_id": job_id,
        "type": job["type"],
        "status": job["status"],
        "created_at": job["created_at"]
    }
    
    # Add optional fields if they exist
    if "start_time" in job:
        response["start_time"] = job["start_time"]
    if "end_time" in job:
        response["end_time"] = job["end_time"]
        response["duration"] = job["end_time"] - job["start_time"]
    if "return_code" in job:
        response["return_code"] = job["return_code"]
    
    # Get last 20 lines of logs for quick preview
    if "logs" in job and job["logs"]:
        log_lines = job["logs"].splitlines()
        response["log_preview"] = "\n".join(log_lines[-20:])
    
    return jsonify(response)

@app.route("/jobs/<job_id>/logs", methods=["GET"])
def get_job_logs(job_id):
    """Get the full logs for a job."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify({
        "job_id": job_id,
        "logs": jobs[job_id].get("logs", "")
    })

if __name__ == "__main__":
    # Ensure the src directory exists
    if not SRC_DIR.exists():
        print(f"Error: Source directory {SRC_DIR} does not exist!")
        sys.exit(1)
    
    # Create any required output directories
    os.makedirs(ROOT_DIR / "models", exist_ok=True)
    os.makedirs(ROOT_DIR / "logs", exist_ok=True)
    
    # Print setup info
    print(f"API starting with:")
    print(f"  - Root directory: {ROOT_DIR}")
    print(f"  - Source directory: {SRC_DIR}")
    print(f"  - Python interpreter: {PYTHON_EXE}")
    
    # Start the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)