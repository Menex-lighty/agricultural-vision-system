from flask import Flask, jsonify, request
import pandas as pd
import csv
import os
from datetime import datetime
import threading
import time
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the directory path first
directory_path = "C:/Users/rishu/Desktop/tifan"

# Ensure the directory exists
os.makedirs(directory_path, exist_ok=True)

# Function to get the latest CSV file
def get_latest_csv_file(directory_path):
    """Find the most recent CSV file in the directory"""
    try:
        # List all CSV files in directory
        csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
                    if f.endswith('.csv') and f.startswith('sapling_data_')]
       
        if not csv_files:
            print(f"No sapling data CSV files found in {directory_path}")
            return os.path.join(directory_path, f"sapling_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
           
        # Get the most recently modified file
        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"Using latest CSV file: {latest_file}")
        return latest_file
    except Exception as e:
        print(f"Error finding latest CSV file: {e}")
        return os.path.join(directory_path, f"sapling_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# Initialize CSV_PATH - will be updated in load_csv_data if needed
CSV_PATH = get_latest_csv_file(directory_path)

# Lock for thread-safe operations
LOCK = threading.Lock()

# Global data storage
stats = {
    "properly_planted": 0,
    "improperly_planted": 0,
    "total": 0,
    "last_detection_time": None,
    "last_status": None
}

sapling_data = []

def load_csv_data():
    """Load data from CSV file and update global stats"""
    global stats, sapling_data, CSV_PATH
   
    try:
        # Get the latest CSV file
        latest_csv = get_latest_csv_file(directory_path)
        if latest_csv is not None:
            CSV_PATH = latest_csv
       
        if not os.path.exists(CSV_PATH):
            print(f"Warning: CSV file not found at {CSV_PATH}")
            # Create an empty CSV with headers if it doesn't exist
            with open(CSV_PATH, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'frame_number', 'sapling_id', 'plantation_status', 'angle', 'confidence', 'position'])
            return
           
        # Read the CSV file
        df = pd.read_csv(CSV_PATH)
       
        with LOCK:
            # Convert DataFrame to list of dictionaries for API
            sapling_data = df.to_dict('records')
           
            # Update statistics
            properly_planted = df[df['plantation_status'] == 'Properly Planted'].shape[0]
            improperly_planted = df[df['plantation_status'] == 'Tilted'].shape[0]
            total = df.shape[0]
           
            # Get the last detection time
            if not df.empty:
                last_row = df.iloc[-1]
                last_time = last_row['timestamp']
                last_status = last_row['plantation_status'] == 'Properly Planted'
            else:
                last_time = None
                last_status = None
               
            # Update stats dictionary
            stats = {
                "properly_planted": properly_planted,
                "improperly_planted": improperly_planted,
                "total": total,
                "last_detection_time": last_time,
                "last_status": last_status
            }
           
        print(f"Loaded {len(sapling_data)} records from CSV. Properly planted: {properly_planted}, Tilted: {improperly_planted}")
    except Exception as e:
        print(f"Error loading CSV data: {e}")

def csv_monitor():
    """Background thread to monitor CSV file for changes"""
    global CSV_PATH
    last_modified = 0
   
    while True:
        try:
            # First check if there's a newer CSV file than the one we're monitoring
            latest_csv = get_latest_csv_file(directory_path)
            if latest_csv is not None and latest_csv != CSV_PATH:
                print(f"Found newer CSV file: {latest_csv}")
                CSV_PATH = latest_csv
                load_csv_data()
                last_modified = os.path.getmtime(CSV_PATH)
                continue
           
            # Check if current file exists and has been modified
            if os.path.exists(CSV_PATH):
                current_modified = os.path.getmtime(CSV_PATH)
               
                if current_modified > last_modified:
                    print(f"CSV file changed, reloading data...")
                    load_csv_data()
                    last_modified = current_modified
           
            # Sleep for a short time before checking again
            time.sleep(1)
        except Exception as e:
            print(f"Error in CSV monitor: {e}")
            time.sleep(5)  # Wait longer on error

def generate_sample_csv():
    """Generate sample CSV data with 20 saplings in 10 columns"""
    global CSV_PATH
   
    # Create sample data
    sample_data = []
   
    # Generate 20 saplings (10 columns with 2 saplings each)
    for i in range(1, 21):
        # Even IDs are left position, odd IDs are right position
        position = "Left" if i % 2 == 0 else "Right"
       
        # Randomly determine if properly planted (70% chance)
        is_properly_planted = random.random() < 0.7
        status = "Properly Planted" if is_properly_planted else "Tilted"
       
        # Generate appropriate angle based on status
        if is_properly_planted:
            angle = random.uniform(80, 100)  # Properly planted angles
        else:
            # Either too tilted left or right
            angle = random.uniform(30, 50) if random.random() < 0.5 else random.uniform(130, 150)
       
        # Random confidence between 0.6 and 0.98
        confidence = random.uniform(0.6, 0.98)
       
        # Create timestamp with slight time differences
        timestamp = (datetime.now() - pd.Timedelta(seconds=random.randint(1, 300))).strftime('%Y-%m-%d %H:%M:%S')
       
        sample_data.append({
            'timestamp': timestamp,
            'frame_number': i,
            'sapling_id': i,
            'plantation_status': status,
            'angle': round(angle, 2),
            'confidence': round(confidence, 2),
            'position': position
        })
   
    # Sort by sapling_id
    sample_data.sort(key=lambda x: x['sapling_id'])
   
    # Create a new file for the sample data
    CSV_PATH = os.path.join(directory_path, f"sapling_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
   
    # Write to CSV
    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sample_data[0].keys())
        writer.writeheader()
        writer.writerows(sample_data)
   
    print(f"Generated sample CSV with {len(sample_data)} saplings at {CSV_PATH}")
   
    # Reload data
    load_csv_data()

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """API endpoint to get current statistics"""
    with LOCK:
        return jsonify(stats)

@app.route('/api/saplings', methods=['GET'])
def get_saplings():
    """API endpoint to get all sapling data"""
    with LOCK:
        return jsonify(sapling_data)

@app.route('/api/simulate_detection', methods=['GET'])
def simulate_detection():
    """Simulate a new detection by adding a row to the CSV"""
    global CSV_PATH
   
    try:
        # Ensure CSV_PATH exists
        if not os.path.exists(CSV_PATH):
            # Write header only
            with open(CSV_PATH, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'frame_number', 'sapling_id', 'plantation_status', 'angle', 'confidence', 'position'])
            existing_data = []
        else:
            # Read existing data
            with open(CSV_PATH, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                existing_data = list(reader)
               
        # Get next sapling ID
        next_id = 1
        if existing_data:
            try:
                next_id = max([int(row['sapling_id']) for row in existing_data if row['sapling_id'].isdigit()]) + 1
            except:
                next_id = len(existing_data) + 1
               
        # Determine position (alternating)
        position = "Left" if next_id % 2 == 0 else "Right"
       
        # Randomly determine if properly planted (70% chance)
        is_properly_planted = random.random() < 0.7
        status = "Properly Planted" if is_properly_planted else "Tilted"
       
        # Generate appropriate angle based on status
        if is_properly_planted:
            angle = random.uniform(80, 100)  # Properly planted angles
        else:
            # Either too tilted left or right
            angle = random.uniform(30, 50) if random.random() < 0.5 else random.uniform(130, 150)
       
        # Create new detection entry
        new_detection = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'frame_number': len(existing_data) + 1,
            'sapling_id': next_id,
            'plantation_status': status,
            'angle': round(angle, 2),
            'confidence': round(random.uniform(0.7, 0.95), 2),
            'position': position
        }
       
        # Append to CSV file
        with open(CSV_PATH, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=new_detection.keys())
            # Write header if file is empty
            if os.stat(CSV_PATH).st_size == 0:
                writer.writeheader()
            writer.writerow(new_detection)
       
        # Reload data
        load_csv_data()
       
        return jsonify({"status": "success", "message": f"Added new detection (ID: {next_id})"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/reset', methods=['GET'])
def reset_data():
    """Reset data by generating new sample data"""
    try:
        generate_sample_csv()
        return jsonify({"status": "success", "message": "Reset data with sample saplings"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Simple index page"""
    return """
    <html>
        <head>
            <title>Sapling Monitoring Server</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #2c3e50; }
                .endpoint { background: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }
                code { background: #eee; padding: 2px 5px; border-radius: 3px; }
                button { padding: 8px 16px; margin: 10px 0; background: #3498db; color: white; border: none;
                         border-radius: 4px; cursor: pointer; }
                button:hover { background: #2980b9; }
            </style>
            <script>
                function callApi(endpoint) {
                    fetch(endpoint)
                        .then(response => response.json())
                        .then(data => {
                            alert(data.message || JSON.stringify(data));
                        })
                        .catch(error => {
                            alert('Error: ' + error);
                        });
                }
            </script>
        </head>
        <body>
            <h1>Sapling Monitoring Server</h1>
           
            <h2>Actions</h2>
            <button onclick="callApi('/api/reset')">Reset with Sample Data</button>
            <button onclick="callApi('/api/simulate_detection')">Simulate New Detection</button>
           
            <h2>API Endpoints</h2>
            <div class="endpoint">
                <code>GET /api/stats</code>: Get current statistics
            </div>
            <div class="endpoint">
                <code>GET /api/saplings</code>: Get all sapling data
            </div>
            <div class="endpoint">
                <code>GET /api/simulate_detection</code>: Simulate a new detection
            </div>
            <div class="endpoint">
                <code>GET /api/reset</code>: Reset data with sample saplings
            </div>
        </body>
    </html>
    """

if __name__ == '__main__':
    # Initial load of CSV data
    load_csv_data()
   
    # Start background thread for monitoring CSV
    monitor_thread = threading.Thread(target=csv_monitor, daemon=True)
    monitor_thread.start()
   
    # Start Flask app - use 0.0.0.0 to make it accessible from other devices
    app.run(host='0.0.0.0', port=5000, debug=True)