from flask import Flask, render_template, jsonify, request
import os
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import json
from anthropic import Anthropic
import sqlite3
import random
from datetime import datetime

app = Flask(__name__)

# Initialize database
def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def get_db():
    db = sqlite3.connect('analysis.db')
    db.row_factory = sqlite3.Row
    return db

def get_next_images():
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "Dataset 2")
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    
    # Group images by their prefix
    prefix_groups = {}
    for img in image_files:
        prefix = img.split('_')[0]
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(img)
    
    # Filter groups that have at least 2 images
    valid_groups = {k: v for k, v in prefix_groups.items() if len(v) >= 2}
    
    if not valid_groups:
        return None
    
    # Randomly select a prefix
    selected_prefix = random.choice(list(valid_groups.keys()))
    
    # Get the images for this prefix and sort them
    prefix_images = valid_groups[selected_prefix]
    prefix_images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Take consecutive pairs
    pairs = [(prefix_images[i], prefix_images[i+1]) for i in range(0, len(prefix_images)-1, 2)]
    
    # If we've used all pairs, start over from the beginning
    if not pairs:
        return [prefix_images[0], prefix_images[1]]
    
    # Return a random pair
    return random.choice(pairs)

def get_anthropic_analysis(img1_name, img2_name, count):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: Missing ANTHROPIC_API_KEY in environment"
    
    client = Anthropic(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content":
                f"As an environmental monitoring expert, analyze these satellite imagery changes:\n"
                f"Land type: {img1_name.split('_')[0]}\n"
                f"Changes detected: {count}\n\n"
                "Provide analysis in exactly this format:\n"
                "Environmental Impact:\n<1-2 sentences on the significance of changes to the environment, urban development, or land use>\n\n"
                "Potential Concerns:\n<1-2 sentences on implications for sustainability, conservation, or community impact>\n\n"
                "Suggested Monitoring:\n<1-2 bullet points on recommended environmental or urban planning follow-up actions>"
        },
    ]
    
    resp = client.messages.create(
        model="claude-3-opus-20240229",
        messages=messages,
        max_tokens=300,
    )
    
    return resp.content[0].text.strip()

def analyze_history():
    db = get_db()
    history = db.execute('SELECT * FROM analysis_history ORDER BY timestamp').fetchall()
    
    if not history:
        return "No history available for analysis."
    
    analysis_data = []
    for row in history:
        analysis_data.append({
            'timestamp': row['timestamp'],
            'area_type': row['image1_name'].split('_')[0],
            'change_count': row['change_count']
        })
    
    summary = "ENVIRONMENTAL MONITORING REPORT\n\n"
    for data in analysis_data:
        summary += f"- {data['timestamp']}: {data['area_type']} area - {data['change_count']} changes\n"
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: Missing ANTHROPIC_API_KEY in environment"
    
    client = Anthropic(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": f"{summary}\n\n"
                      "Based on this environmental monitoring history, provide analysis in exactly this format:\n"
                      "Environmental Impact:\n<2-3 sentences on overall patterns and their significance for the environment>\n\n"
                      "Potential Concerns:\n<2-3 sentences on long-term implications for sustainability and conservation>\n\n"
                      "Suggested Monitoring:\n<2-3 bullet points on recommended environmental assessment and planning actions>"
        },
    ]
    
    resp = client.messages.create(
        model="claude-3-opus-20240229",
        messages=messages,
        max_tokens=500,
    )
    
    return resp.content[0].text.strip()

def load_and_process_images(img1_name, img2_name):
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "Dataset 2")
    
    # Load images
    img1_path = os.path.join(dataset_path, img1_name)
    img2_path = os.path.join(dataset_path, img2_name)
    
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    
    # Convert to numpy arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    # Compute difference
    diff = np.abs(img1_array - img2_array).sum(axis=2)
    threshold = 50
    mask = (diff > threshold).astype(np.uint8) * 255
    
    # Create difference visualization with yellow and brown
    diff_img = Image.new('RGB', img1.size, (139, 69, 19))  # Brown background
    yellow_overlay = Image.new('RGB', diff_img.size, (255, 255, 0))  # Yellow for differences
    
    # Create mask for overlay
    mask_img = Image.fromarray(mask)
    
    # Composite the images
    diff_img.paste(yellow_overlay, mask=mask_img)
    
    # Convert images to base64 for web display
    def img_to_base64(img):
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    img1_data = img_to_base64(img1)
    img2_data = img_to_base64(img2)
    diff_data = img_to_base64(diff_img)
    change_count = np.sum(mask > 0)
    
    # Get Anthropic analysis
    analysis = get_anthropic_analysis(img1_name, img2_name, change_count)
    
    # Store in database
    db = get_db()
    db.execute(
        'INSERT INTO analysis_history (image1_name, image2_name, image1_data, image2_data, diff_data, analysis_text, change_count) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (img1_name, img2_name, img1_data, img2_data, diff_data, analysis, int(change_count))
    )
    db.commit()
    
    return {
        'img1': img1_data,
        'img2': img2_data,
        'diff': diff_data,
        'img1_name': img1_name,
        'img2_name': img2_name,
        'analysis': analysis,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_images')
def get_images():
    selected_images = get_next_images()
    if not selected_images:
        return jsonify({'error': 'No valid image pairs found'}), 400
    
    result = load_and_process_images(selected_images[0], selected_images[1])
    return jsonify(result)

@app.route('/get_history')
def get_history():
    db = get_db()
    history = db.execute('SELECT * FROM analysis_history ORDER BY timestamp DESC').fetchall()
    return jsonify([{
        'id': row['id'],
        'timestamp': row['timestamp'],
        'img1_name': row['image1_name'],
        'img2_name': row['image2_name'],
        'img1': row['image1_data'],
        'img2': row['image2_data'],
        'diff': row['diff_data'],
        'analysis': row['analysis_text']
    } for row in history])

@app.route('/analyze_history')
def get_history_analysis():
    analysis = analyze_history()
    return jsonify({'analysis': analysis})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5009)
