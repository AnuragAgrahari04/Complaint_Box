# main.py
from flask import Flask, request, jsonify
import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Initialize AI Models and API Clients ---

# 1. Groq Client for LLaMA3
# It will read the key from your .env file
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Make sure it's in your .env file.")
    client = Groq(api_key=groq_api_key)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    client = None

# 2. Salesforce BLIP for Image Captioning
try:
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("BLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading BLIP model: {e}")
    processor = None
    model = None


# --- Helper Functions ---

def analyze_complaint_text(text):
    """
    Analyzes the complaint text using LLaMA3 via Groq API.
    """
    if not client:
        return {"error": "Groq client is not initialized."}

    system_prompt = """
    You are an expert complaint analysis system. Your task is to analyze the user's complaint and return a structured JSON object with the following fields:
    - "urgency": (Enum: "High", "Medium", "Low")
    - "department": (e.g., "Public Works", "Sanitation", "Traffic", "Health Department", "Billing", "Customer Service")
    - "category": (A specific category within the department, e.g., "Potholes", "Garbage Collection", "Broken Streetlight", "Billing Dispute")
    - "subcategory": (A more detailed subcategory, e.g., "Main Street Pothole", "Missed weekly pickup", "Flickering light", "Incorrect invoice amount")
    - "summary": (A brief, one-sentence summary of the complaint)

    Analyze the user's text and provide the most logical classification. Do not add any explanations or introductory text outside of the JSON object.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Analyze the following complaint: \"{text}\"",
                },
            ],
            model="llama3-8b-8192",
            temperature=0.2,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during Groq API call: {e}")
        return {"error": "Failed to analyze complaint with LLM.", "details": str(e)}


def caption_image(image_file):
    """
    Generates a caption for an image using the BLIP model.
    """
    if not processor or not model:
        return {"error": "BLIP model is not initialized."}

    try:
        raw_image = Image.open(image_file).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error during image captioning: {e}")
        return {"error": "Failed to generate image caption.", "details": str(e)}


# --- API Endpoints ---

@app.route('/')
def home():
    """A simple welcome message to verify the API is running."""
    return "Welcome to the Complaint Assistant API!"


@app.route('/analyze/text', methods=['POST'])
def handle_text_complaint():
    """
    Endpoint to analyze a text-based complaint.
    """
    if not request.is_json:
        return jsonify({"error": "Invalid request: payload must be JSON."}), 400

    data = request.get_json()
    complaint_text = data.get('complaint_text')

    if not complaint_text:
        return jsonify({"error": "Missing 'complaint_text' in request body."}), 400

    analysis_result = analyze_complaint_text(complaint_text)

    if isinstance(analysis_result, dict) and "error" in analysis_result:
        return jsonify(analysis_result), 500

    return analysis_result, 200, {'Content-Type': 'application/json'}


@app.route('/analyze/image', methods=['POST'])
def handle_image_complaint():
    """
    Endpoint to analyze an image-based complaint.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No 'image' file found in the request."}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    caption = caption_image(image_file)
    if isinstance(caption, dict) and "error" in caption:
        return jsonify(caption), 500

    analysis_result_str = analyze_complaint_text(caption)
    if isinstance(analysis_result_str, dict) and "error" in analysis_result_str:
        return jsonify(analysis_result_str), 500

    import json
    final_response = json.loads(analysis_result_str)
    final_response['generated_caption'] = caption

    return jsonify(final_response), 200


# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
