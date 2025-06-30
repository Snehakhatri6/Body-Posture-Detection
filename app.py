from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import hashlib
import secrets
import os
from model import analyze_squat

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        );
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.form
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    if not all([name, email, password]):
        return jsonify({'success': False, 'message': 'Missing fields'}), 400
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                     (name, email, hash_password(password)))
        conn.commit()
        return jsonify({'success': True, 'message': 'User registered'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'message': 'Email already exists'}), 409
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.form
    email = data.get('email')
    password = data.get('password')
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?',
                        (email, hash_password(password))).fetchone()
    conn.close()
    if user:
        # Generate a random session token (for demo; not stored server-side)
        token = secrets.token_hex(16)
        return jsonify({'success': True, 'message': 'Login successful', 'token': token})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    data = request.form
    email = data.get('email')
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    if user:
        # In real apps, email a reset link. Here, just return a message.
        return jsonify({'success': True, 'message': 'Password reset link sent (simulated).'})
    else:
        return jsonify({'success': False, 'message': 'Email not found'}), 404

@app.route('/api/upload-exercise', methods=['POST'])
def upload_exercise():
    if 'video' not in request.files:
        print("No video uploaded in request.")
        return jsonify({'success': False, 'message': 'No video uploaded'}), 400
    video = request.files['video']
    input_path = os.path.join(UPLOAD_FOLDER, video.filename)
    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{video.filename}")
    try:
        video.save(input_path)
        print(f"Saved input video to {input_path}")
        # Analyze and save processed video
        output_path = os.path.join(PROCESSED_FOLDER, f"processed_{video.filename}")
        analyze_squat(input_path, output_path)
        # Check if output file exists and is not empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            print(f"Processed video not created or too small: {output_path}")
            return jsonify({'success': False, 'message': 'Processing failed or output video is invalid.'}), 500
        print(f"Processed video saved to {output_path}")
        return jsonify({'success': True, 'output_video': f"/processed/{os.path.basename(output_path)}"}), 200
    except Exception as e:
        print(f"Error during video processing: {e}")
        return jsonify({'success': False, 'message': f'Error during processing: {str(e)}'}), 500

@app.route('/processed/<filename>')
def processed_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/exercise.html')
def serve_exercise_html():
    return send_from_directory('static', 'exercise.html')

@app.route('/')
def home():
    return "Flask is running!"

if __name__ == '__main__':
    create_users_table()
    app.run(debug=True)