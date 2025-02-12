from openai import OpenAI
import streamlit as st
import base64
from PIL import Image
import json
import cv2

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["API_KEY"],  # Make sure this matches your secrets.toml
)

def analyze_image(image_path):
    """Send image to LLM via OpenRouter.ai"""
    base64_image = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
    
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "http://localhost:8501",  # Update with your deployment URL
            "X-Title": "Vehicle Analysis App",
        },
        model="google/gemini-2.0-flash-exp:free",
        messages=[
            {
                "role": "system",
                "content": """Return ONLY a JSON object wrapped in ```json code blocks with these EXACT keys:
                {
                    "car_model": "string", 
                    "car_color": "string",
                    "number_plate": "string"
                }
                No extra text or explanations."""
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
        max_tokens=300
    )
    
    try:
        response_text = completion.choices[0].message.content
        print(response_text)
        
        # Clean response text before parsing
        cleaned_response = response_text.strip().replace('```json', '').replace('```', '')
        
        # Add debug logging
        st.write("Raw LLM Response:", response_text)  # For debugging
        
        parsed_response = json.loads(cleaned_response)
        return parsed_response
        
    except json.JSONDecodeError:
        # Try to find JSON substring if response contains extra text
        try:
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1
            parsed_response = json.loads(cleaned_response[json_start:json_end])
            return parsed_response
        except:
            st.error(f"Invalid JSON format in response: {cleaned_response}")
            return {}
            
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return {}

# App UI
st.title("Vehicle Analysis System")

# Image upload section
uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Vehicle", use_container_width=True)
    
    # Save uploaded file
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Analysis section
    if st.button("Analyze Vehicle"):
        with st.spinner("Analyzing..."):
            try:
                result = analyze_image("temp_image.jpg")
                
                # Display results in boxes
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
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Video upload section
elif input_type == "Video Upload":
    video_file = st.file_uploader("Upload Vehicle Video", type=["mp4", "avi", "mov"])
    if video_file:
        # Create temporary video file
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())
        
        # Initialize video capture
        cap = cv2.VideoCapture("temp_video.mp4")
        
        if not cap.isOpened():
            st.error("Failed to open video file")
        else:
            # Extract first frame
            success, frame = cap.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                st.session_state.current_frame = pil_image.tobytes()
                st.image(pil_image, caption="First Video Frame", use_column_width=True)
            else:
                st.error("Could not read frames from video")
            cap.release() 