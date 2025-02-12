import streamlit as st
import base64
import json
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
import ffmpeg
import time
import tempfile
from typing import Union
import os
import io
import requests


# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["API_KEY"],
)

def analyze_image(image):
    """Send image to LLM via OpenRouter.ai"""
    try:
        # Handle both bytes and file-like objects
        buffered = image if isinstance(image, bytes) else image.getvalue()
        base64_image = base64.b64encode(buffered).decode("utf-8")
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Vehicle Analysis App",
            },
            model="google/gemini-2.0-flash-exp:free",
            messages=[
                {
                    "role": "system",
                    "content": """Return ONLY a JSON object with these EXACT keys:
                    {
                        "car_model": "string", 
                        "car_color": "string",
                        "number_plate": "string"
                    }
                    No extra text."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            timeout=30  # Add timeout
        )
        
        response_text = completion.choices[0].message.content
        cleaned_response = response_text.strip().replace('```json', '').replace('```', '')
        parsed_response = json.loads(cleaned_response)
        return parsed_response if isinstance(parsed_response, dict) else {}
    
    except requests.exceptions.Timeout:
        st.error("Analysis timed out (over 30 seconds)")
        return {}
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {}

# Session state initialization
if 'input_type' not in st.session_state:
    st.session_state.input_type = "image"
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None

st.title("Vehicle Analysis System")

# Input type selection
input_type = st.radio("Select Input Type:", 
                     ["Image Upload", "Camera", "Video Upload"],
                     horizontal=True)

def process_video(uploaded_file) -> Union[bytes, None]:
    """Process video file with error handling and performance tracking"""
    try:
        start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            video_path = tmp_file.name

        # Validate video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or frame_count == 0:
            st.error("Invalid video file: Unable to read video properties")
            return None

        duration = frame_count / fps
        if duration > 30:  # Limit to 30-second videos
            st.error("Video too long (max 30 seconds)")
            return None

        # Process middle frame for better context
        target_frame = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        progress = st.progress(0)
        status_text = st.empty()
        
        success, frame = cap.read()
        if not success:
            status_text.error("Failed to read video frames")
            return None

        progress.progress(50)
        status_text.text("Converting frame...")
        
        # Convert BGR to RGB and create PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        frame_bytes = img_byte_arr.getvalue()
        
        progress.progress(100)
        status_text.text(f"Processed {duration:.1f}s video in {time.time()-start_time:.1f}s")
        
        return frame_bytes

    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        return None
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if os.path.exists(video_path):
            os.unlink(video_path)

# Handle different input types
if input_type == "Image Upload":
    uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.session_state.current_frame = uploaded_file.getvalue()
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

elif input_type == "Camera":
    camera_file = st.camera_input("Take Vehicle Photo")
    if camera_file:
        st.session_state.current_frame = camera_file.getvalue()
        st.image(camera_file, caption="Captured Image", use_container_width=True)

elif input_type == "Video Upload":
    video_file = st.file_uploader("Upload Vehicle Video", type=["mp4", "avi", "mov"])
    if video_file:
        processed_frame = process_video(video_file)
        if processed_frame:
            st.session_state.current_frame = processed_frame
            st.image(Image.open(io.BytesIO(processed_frame)), 
                    caption="Analyzed Video Frame",
                    use_container_width=True)

# Analysis button section
if st.button("Analyze Vehicle"):
    if st.session_state.get('current_frame') is not None:
        with st.spinner("Analyzing..."):
            try:
                result = analyze_image(st.session_state.current_frame)
                
                if result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info("**Car Model**")
                        st.success(result.get("car_model", "N/A"))
                    
                    with col2:
                        st.info("**Car Color**")
                        st.success(result.get("car_color", "N/A"))
                    
                    with col3:
                        st.info("**Number Plate**")
                        st.success(result.get("number_plate", "N/A"))
                else:
                    st.error("No valid analysis results received")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    else:
        st.warning("Please upload or capture media first")

#Requirements:pip install opencv-python-headless streamlit openai python-dotenv