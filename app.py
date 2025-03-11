from quart import Quart, render_template, request, redirect, url_for, send_file, jsonify, session, flash
import os
import pandas as pd
from datetime import datetime
from paddleocr import PaddleOCR
from google.cloud import speech
from pydub import AudioSegment
import base64
import numpy as np
import cv2
import zipfile
import json
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import dotenv
import socketio
import aiofiles
import hypercorn.asyncio
import asyncio
from hypercorn.config import Config
import aiofiles.os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
import uuid
import time
import glob

# Load environment variables
dotenv.load_dotenv()

# Initialize Quart app
app = Quart(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# Initialize SocketIO
sio = socketio.AsyncServer(async_mode="asgi")
asgi_app = socketio.ASGIApp(sio, app)

# Configure Hypercorn
config = Config()
config.bind = ["0.0.0.0:5000"]

# Admin credentials
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

###################################################### Database Setup ######################################################
# Using SQLAlchemy and async SQLite engine
DATABASE_URI = "sqlite+aiosqlite:///data/inventory.db"

# Create an async SQLAlchemy engine
async_engine = create_async_engine(DATABASE_URI, echo=False)

# Define Base
Base = declarative_base()

# Create an async session factory
AsyncSessionFactory = sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession
)

# Define an async session dependency
async def get_db_session():
    async with AsyncSessionFactory() as session:
        yield session

# Define Product model
class Product(Base):
    __tablename__ = "products"
    
    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String(255), nullable=False)
    brand = sa.Column(sa.String(255), nullable=True)
    
###################################################### Constants ######################################################
LOCATIONS_FILE = "locations.json"
INVENTORY_STATUS_FILE = "inventory_status.json"

# Resize image constants
MAX_WIDTH = 640
MAX_HEIGHT = 480

PRODUCTS_DF = None

###################################################### Utility Functions ######################################################
async def ensure_inventory_directory():
    """Ensure the inventories directory exists."""
    if not os.path.exists("inventories"):
        await aiofiles.os.makedirs("inventories", exist_ok=True)

@app.before_serving
async def ensure_temp_audio_directory():
    """Ensure temp_audio directory exists."""
    temp_dir = "temp_audio"
    if not os.path.exists(temp_dir):
        await aiofiles.os.makedirs(temp_dir, exist_ok=True)
    print(f"Temporary audio directory ready: {temp_dir}")

async def load_locations():
    """Load or create the locations file asynchronously."""
    if not os.path.exists(LOCATIONS_FILE):
        default_locations = ["Location 1"]
        async with aiofiles.open(LOCATIONS_FILE, "w", encoding="utf-8") as f:
            await f.write(json.dumps(default_locations, indent=4))

    async with aiofiles.open(LOCATIONS_FILE, "r", encoding="utf-8") as f:
        content = await f.read()
        return json.loads(content)

async def read_excel_async(filename):
    """Read Excel asynchronously using Pandas"""
    if await aiofiles.os.path.exists(filename):
        df = await asyncio.to_thread(lambda: pd.read_excel(filename))
        return df
    return None

async def write_excel_async(df, filename):
    """Write DataFrame to Excel asynchronously using Pandas"""
    await asyncio.to_thread(lambda: df.to_excel(filename, index=False, engine="openpyxl"))

async def load_inventory_status():
    """Ensure inventory_status.json exists and load inventory status asynchronously."""
    if not os.path.exists(INVENTORY_STATUS_FILE):
        await save_inventory_status(True)

    try:
        async with aiofiles.open(INVENTORY_STATUS_FILE, "r") as f:
            content = await f.read()
            return json.loads(content).get("inventory_enabled", True)
    except (json.JSONDecodeError, FileNotFoundError):
        return True

async def save_inventory_status(status):
    """Save the inventory status asynchronously."""
    async with aiofiles.open(INVENTORY_STATUS_FILE, "w") as f:
        await f.write(json.dumps({"inventory_enabled": status}))

async def export_all_inventories():
    """Creates a ZIP file of all inventory Excel files and offers it for download."""
    zip_filename = "inventories/all_inventories.zip"

    def create_zip():
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            for filename in os.listdir("inventories"):
                if filename.endswith(".xlsx"):
                    zipf.write(os.path.join("inventories", filename), filename)

    await asyncio.to_thread(create_zip)
    return await send_file(zip_filename, as_attachment=True)

async def reset_all_inventories():
    """Deletes all inventory files asynchronously."""
    def remove_files():
        for file in os.listdir("inventories"):
            if file.endswith(".xlsx"):
                os.remove(os.path.join("inventories", file))
    
    await asyncio.to_thread(remove_files)

###################################################### Initialize Database ######################################################
async def init_database():
    """Initialize the database tables and load product data if needed."""
    await ensure_inventory_directory()
    
    # Create tables asynchronously
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Use async session for queries
    async with AsyncSessionFactory() as session:
        first_product = (await session.execute(select(Product))).scalars().first()
        
        # If no products, load from Excel
        if first_product is None and os.path.exists("data/Database.xlsx"):
            try:
                df = pd.read_excel("data/Database.xlsx")
                for _, row in df.iterrows():
                    product = Product(name=row["Product"], brand=row.get("Brand", ""))
                    session.add(product)
                await session.commit()
                print(f"Loaded {len(df)} products from Database.xlsx")
            except Exception as e:
                print(f"Error loading products: {e}")
                await session.rollback()

# Initialize the app at startup
@app.before_serving
async def before_serving():
    await init_database()
    app.locations = await load_locations()
    print("Application initialized successfully")

@app.before_serving
async def load_product_data():
    """Load product data from Excel file at startup."""
    global PRODUCTS_DF
    
    def load_df():
        df = pd.read_excel("data/Database.xlsx")
        # Convert product names to lowercase for case-insensitive searching
        df["Product"] = df["Product"].astype(str).str.lower()
        return df
    
    PRODUCTS_DF = await asyncio.to_thread(load_df)
    print(f"Loaded {len(PRODUCTS_DF)} products from Excel file")

# Before_serving functions
@app.before_serving
async def cleanup_startup():
    """Clean up any debug files at startup."""
    debug_dir = "debug"
    if os.path.exists(debug_dir):
        try:
            # Get all jpg files in the debug directory
            debug_files = glob.glob(os.path.join(debug_dir, "*.jpg"))
            
            # Delete all files
            for file_path in debug_files:
                try:
                    os.remove(file_path)
                    print(f"Startup cleanup: Removed debug file {file_path}")
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            
            print(f"Startup cleanup: Removed {len(debug_files)} debug files")
        except Exception as e:
            print(f"Error in startup cleanup: {e}")

@app.before_serving
async def setup_debug_cleanup():
    """Setup background task to clean up old debug files periodically."""
    async def cleanup_old_debug_files():
        while True:
            try:
                debug_dir = "debug"
                if os.path.exists(debug_dir):
                    # Get all jpg files in the debug directory
                    debug_files = glob.glob(os.path.join(debug_dir, "*.jpg"))
                    
                    # Get current time
                    current_time = time.time()
                    
                    # Check each file
                    for file_path in debug_files:
                        try:
                            # Get file modification time
                            file_mod_time = os.path.getmtime(file_path)
                            
                            # Delete files older than 5 minutes
                            if current_time - file_mod_time > 300:
                                os.remove(file_path)
                                print(f"Removed old debug file: {file_path}")
                        except Exception as e:
                            print(f"Error checking file {file_path}: {e}")
                    
                    # Report total number of files remaining
                    remaining_files = glob.glob(os.path.join(debug_dir, "*.jpg"))
                    if remaining_files:
                        print(f"There are still {len(remaining_files)} debug files remaining")
                    
            except Exception as e:
                print(f"Error in debug cleanup task: {e}")
            
            # Run every 5 minutes
            await asyncio.sleep(300)
    
    # Start the background task
    asyncio.create_task(cleanup_old_debug_files())

@app.before_serving
async def setup_audio_cleanup():
    """Setup background task to clean up old audio files periodically."""
    async def cleanup_old_audio_files():
        while True:
            try:
                temp_dir = "temp_audio"
                if os.path.exists(temp_dir):
                    # Get all files in the temp_audio directory
                    audio_files = glob.glob(os.path.join(temp_dir, "*.*"))
                    
                    # Get current time
                    current_time = time.time()
                    
                    # Check each file
                    for file_path in audio_files:
                        try:
                            # Get file modification time
                            file_mod_time = os.path.getmtime(file_path)
                            
                            # Delete files older than 5 minutes
                            if current_time - file_mod_time > 300:
                                os.remove(file_path)
                                print(f"Removed old audio file: {file_path}")
                        except Exception as e:
                            print(f"Error checking audio file {file_path}: {e}")
                    
                    # Report total number of files remaining
                    remaining_files = glob.glob(os.path.join(temp_dir, "*.*"))
                    if remaining_files:
                        print(f"There are still {len(remaining_files)} audio files remaining")
                    
            except Exception as e:
                print(f"Error in audio cleanup task: {e}")
            
            # Run every 5 minutes
            await asyncio.sleep(300)
    
    # Start the background task
    asyncio.create_task(cleanup_old_audio_files())


###################################################### Speech Recognition ######################################################
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not credentials_path:
    print("WARNING: GOOGLE_APPLICATION_CREDENTIALS is not set. Speech recognition will not work.")
    client = None
else:
    client = speech.SpeechClient()

async def convert_audio_to_wav(audio_path, request_id):
    """Convert any audio format to 16kHz WAV asynchronously."""
    converted_path = f"temp_audio/converted_{request_id}.wav"

    def process_audio():
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(converted_path, format="wav")

    await asyncio.to_thread(process_audio)
    return converted_path

async def transcribe_audio_google(audio_path):
    """Send recorded audio to Google Speech-to-Text API asynchronously."""
    if client is None:
        return ""
    
    try:
        # Read the file with aiofiles
        async with aiofiles.open(audio_path, "rb") as audio_file:
            content = await audio_file.read()
        
        # Run the recognition in a thread to avoid blocking
        def recognize():
            try:
                # Create recognition config
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # WAV format
                    sample_rate_hertz=16000,
                    language_code="ro-RO",  # Romanian language
                    model="default",
                    use_enhanced=True,
                )
                
                # Create audio object with bytes content
                audio = speech.RecognitionAudio(content=content)
            
                response = client.recognize(config=config, audio=audio)
                
                print(f"Google STT response: {response}")
                
                if response.results and response.results[0].alternatives:
                    return response.results[0].alternatives[0].transcript
                return ""
            except Exception as e:
                print(f"Error in Google recognize(): {e}")
                import traceback
                traceback.print_exc()
                return ""
                
        result = await asyncio.to_thread(recognize)
        return result
        
    except Exception as e:
        print(f"Error in transcribe_audio_google: {e}")
        import traceback
        traceback.print_exc()
        return ""


###################################################### OCR Functions ######################################################
# Initialize PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    use_gpu=False,
    precision='fp16',
    det_limit_side_len=640,
    det_db_thresh=0.5,
    det_db_box_thresh=0.5,
    det_db_unclip_ratio=1.3,
    rec_batch_num=5,
    total_process_num=2
)

async def perform_ocr(frame, unique_id, debug_files):
    """Perform OCR on the given frame and return text blocks."""
    if frame is None:
        print("Frame is None, cannot perform OCR")
        return []

    try:
        # Ensure the debug directory exists
        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)

        # Save debug images with unique filenames
        debug_frame_path = os.path.join(debug_dir, f"debug_frame_{unique_id}.jpg")
        cv2.imwrite(debug_frame_path, frame)
        debug_files.append(debug_frame_path)
        print(f"Saved OCR debug frame to {debug_frame_path}")

        # Run OCR in a separate thread
        def run_ocr():
            try:
                return ocr.ocr(frame, cls=False)
            except Exception as e:
                print(f"OCR engine error: {e}")
                return []

        result = await asyncio.to_thread(run_ocr)

        print(f"OCR result type: {type(result)}")

        if result is None or not isinstance(result, list) or not result:
            print("OCR returned empty result")
            return []

        # Process OCR results
        h, w, _ = frame.shape
        text_blocks = []

        for line in result:
            if not line:
                continue

            for word in line:
                if not word or len(word) < 2:
                    continue

                try:
                    points, (text, confidence) = word

                    # Ensure valid data
                    if not points or not text:
                        continue

                    print(f"Detected text: '{text}' with confidence {confidence}")

                    # Calculate bounding box
                    x_min = min(point[0] for point in points)
                    y_min = min(point[1] for point in points)
                    x_max = max(point[0] for point in points)
                    y_max = max(point[1] for point in points)

                    text_blocks.append({
                        "text": text,
                        "confidence": float(confidence),
                        "x": float(x_min / w),
                        "y": float(y_min / h),
                        "width": float((x_max - x_min) / w),
                        "height": float((y_max - y_min) / h)
                    })
                except Exception as word_error:
                    print(f"Error processing word: {word_error}")
                    continue

        print(f"Total text blocks detected: {len(text_blocks)}")
        return text_blocks

    except Exception as e:
        print(f"Exception in OCR: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

###################################################### Socket.IO Handlers ######################################################
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.on("capture")
async def handle_capture(sid, data):
    """Handle image capture from frontend, run OCR, and return results asynchronously."""
    # Initialize debug files
    debug_files = []
    unique_id = uuid.uuid4().hex
    
    try:
        print(f"Received capture request from {sid}")
        
        # Make sure data is in expected format
        if not isinstance(data, str) or not data.startswith('data:image'):
            await sio.emit("processed_frame", {
                "error": "Invalid image format. Expected data URL."
            }, to=sid)
            return
        
        # Create debug directory if it doesn't exist
        debug_dir = "debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Decode base64 image data
        try:
            image_format = data.split(';')[0].split('/')[1]
            frame_data = base64.b64decode(data.split(',')[1])
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error decoding image: {e}")
            await sio.emit("processed_frame", {
                "error": f"Failed to decode image: {str(e)}"
            }, to=sid)
            return

        if frame is None:
            print("Failed to decode image")
            await sio.emit("processed_frame", {
                "error": "Failed to decode image"
            }, to=sid)
            return

        print(f"Successfully decoded image, shape: {frame.shape}")
        
        # Save a debug copy of the original image
        debug_original_path = os.path.join(debug_dir, f"debug_original_{unique_id}.jpg")
        cv2.imwrite(debug_original_path, frame)
        debug_files.append(debug_original_path)
        print(f"Saved debug image to {debug_original_path}")
        
        # Process the image for improved OCR
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        threshold = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to color for OCR
        processed_frame = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
        
        # Save processed image for debugging
        debug_processed_path = os.path.join(debug_dir, f"debug_processed_{unique_id}.jpg")
        cv2.imwrite(debug_processed_path, processed_frame)
        debug_files.append(debug_processed_path)
        print("Saved processed image for debugging")

        # Run OCR on the processed frame
        text_blocks = await perform_ocr(processed_frame, unique_id, debug_files)
        print(f"OCR found {len(text_blocks)} text blocks")

        # Encode the original frame to send back
        def encode_frame():
            _, buffer = cv2.imencode(f".{image_format}", frame)
            return base64.b64encode(buffer).decode("utf-8")
            
        frame_b64 = await asyncio.to_thread(encode_frame)

        # Emit the processed frame data asynchronously
        await sio.emit("processed_frame", {
            "text_blocks": text_blocks, 
            "image": f"data:image/{image_format};base64,{frame_b64}"
        }, to=sid)
        print(f"Sent response with {len(text_blocks)} text blocks")

        # Clean up debug files
        print(f"Attempting to clean up {len(debug_files)} debug files")
        for file_path in debug_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Successfully cleaned up debug file: {file_path}")
                else:
                    print(f"Debug file not found for cleanup: {file_path}")
            except Exception as cleanup_error:
                print(f"Error cleaning up {file_path}: {cleanup_error}")

        # Double-check cleanup
        remaining_files = [f for f in debug_files if os.path.exists(f)]
        if remaining_files:
            print(f"WARNING: {len(remaining_files)} debug files still remain after cleanup")
        else:
            print("All debug files successfully cleaned up")

    except Exception as e:
        # Clean up files even in case of exception
        print(f"Exception occurred - attempting to clean up {len(debug_files)} debug files")
        for file_path in debug_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up debug file during exception handling: {file_path}")
            except Exception as cleanup_error:
                print(f"Error cleaning up during exception: {cleanup_error}")
        
        print(f"Exception in handle_capture: {str(e)}")
        import traceback
        traceback.print_exc()
        await sio.emit("processed_frame", {
            "error": f"Server error: {str(e)}"
        }, to=sid)

###################################################### Routes ######################################################
@app.route("/process_audio", methods=["POST"])
async def process_audio():
    """Receive audio from frontend, convert it to WAV, and transcribe using Google Speech-to-Text."""
    try:
        # Generate a unique ID for this request
        request_id = uuid.uuid4().hex
        
        files = await request.files
        if "audio" not in files:
            return jsonify({"error": "No audio file"}), 400

        audio_file = files["audio"]
        # Use unique filename for the temp audio blob
        temp_audio_path = f"temp_audio/temp_{request_id}.blob"
        
        # Print audio debug info
        print(f"Received audio file: {audio_file.filename}, MIME type: {audio_file.content_type}")
        
        # Save the file as a binary blob
        audio_data = audio_file.read()
        print(f"Audio data size: {len(audio_data)} bytes")
        
        # Ensure directory exists
        os.makedirs("temp_audio", exist_ok=True)
        
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)
        
        print(f"Saved raw audio data to {temp_audio_path}")
        
        # Create WAV file with unique name
        wav_audio_path = f"temp_audio/converted_{request_id}.wav"
        
        # Create a basic WAV header for 16kHz mono PCM audio
        def create_wav_header(data_size):
            # RIFF header
            header = bytearray()
            header.extend(b'RIFF')
            header.extend((data_size + 36).to_bytes(4, 'little'))  # File size - 8
            header.extend(b'WAVE')
            
            # fmt chunk
            header.extend(b'fmt ')
            header.extend((16).to_bytes(4, 'little'))  # Chunk size
            header.extend((1).to_bytes(2, 'little'))   # Format = PCM
            header.extend((1).to_bytes(2, 'little'))   # Channels = 1 (mono)
            header.extend((16000).to_bytes(4, 'little'))  # Sample rate
            header.extend((32000).to_bytes(4, 'little'))  # Byte rate (SampleRate * NumChannels * BitsPerSample/8)
            header.extend((2).to_bytes(2, 'little'))   # Block align (NumChannels * BitsPerSample/8)
            header.extend((16).to_bytes(2, 'little'))  # Bits per sample
            
            # data chunk
            header.extend(b'data')
            header.extend((data_size).to_bytes(4, 'little'))  # Data size
            
            return header
        
        try:
            # Try using pydub to convert
            print("Attempting to convert with pydub...")
            audio = AudioSegment.from_file(temp_audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(wav_audio_path, format="wav")
            print(f"Successfully converted audio to {wav_audio_path}")
        except Exception as e:
            print(f"Pydub conversion error: {e}")
            # Fallback: create a basic WAV file with a basic header
            try:
                print("Using fallback: creating WAV with basic header")
                with open(wav_audio_path, "wb") as out_file:
                    # Try to create a valid WAV header
                    wav_header = create_wav_header(len(audio_data))
                    out_file.write(wav_header)
                    # Append raw audio data
                    out_file.write(audio_data)
                print(f"Created WAV file with basic header at {wav_audio_path}")
            except Exception as header_error:
                print(f"Header creation error: {header_error}")
                # Last resort: copy the raw data
                with open(wav_audio_path, "wb") as out_file:
                    out_file.write(audio_data)
                print("Last resort: copied raw audio data without processing")

        # Use Google STT for transcription
        try:
            recognized_text = await transcribe_audio_google(wav_audio_path)
            print(f"Transcription result: '{recognized_text}'")
        except Exception as e:
            print(f"Transcription error: {e}")
            recognized_text = ""

        # Cleanup temp files
        try:
            os.remove(temp_audio_path)
            os.remove(wav_audio_path)
            print(f"Cleaned up temp files for request {request_id}")
        except Exception as cleanup_error:
            print(f"Cleanup error for request {request_id}: {cleanup_error}")
            # Continue if cleanup fails

        return jsonify({"recognized_text": recognized_text})
        
    except Exception as e:
        print(f"Process audio error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/ocr")
async def ocr_page():
    """Render OCR scanning page."""
    return await render_template("ocr.html")

@app.route("/admin", methods=["GET", "POST"])
async def admin_page():
    if "admin_logged_in" not in session:
        return redirect(url_for("admin_login"))

    # Load current inventory status
    inventory_status = await load_inventory_status()

    if request.method == "POST":
        form = await request.form
        if "set_inventory" in form:
            inventory_status = form["set_inventory"] == "on"
            await save_inventory_status(inventory_status)

        elif "export_all" in form:
            return await export_all_inventories()

        elif "reset_all" in form:
            await reset_all_inventories()
            flash("Toate inventarele au fost sterse!", "success")

        return redirect(url_for("admin_page"))

    return await render_template("admin.html", inventory_enabled=inventory_status)

@app.route("/admin/login", methods=["GET", "POST"])
async def admin_login():
    if request.method == "POST":
        form = await request.form
        username = form["username"]
        password = form["password"]

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_page"))
        else:
            flash("Username sau parola incorecte", "danger")

    return await render_template("admin_login.html")

@app.route("/admin/logout")
async def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))

@app.route("/")
@app.route("/inventory")
async def index():
    inventory_status = await load_inventory_status()
    locations = await load_locations()
    return await render_template("index.html", inventory_enabled=inventory_status, locations=locations)

@app.route("/inventory/<location_name>", methods=["GET", "POST"])
async def inventory_page(location_name):
    today_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"inventories/{location_name}_inventory_{today_date}.xlsx"

    # Form Submission
    if request.method == "POST":
        form = await request.form
        product_name = form["product_name"]
        cutii = int(form["cutii"])
        fractii = int(form["fractii"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create DataFrame from form submission
        new_data = pd.DataFrame([{
            "Product Name": product_name,
            "Cutii (Boxes)": cutii,
            "Fractii (Fractions)": fractii,
            "Timestamp": timestamp
        }])

        # Append data to the existing file or create a new one
        if await aiofiles.os.path.exists(filename):
            existing_data = await asyncio.to_thread(pd.read_excel, filename)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            updated_data = new_data

        # Save back to Excel
        await asyncio.to_thread(lambda: updated_data.to_excel(filename, index=False))

        return redirect(url_for("inventory_page", location_name=location_name))

    # Load existing inventory data
    inventory_data = []
    existing_products = {}

    df = await read_excel_async(filename)
    if df is not None:
        inventory_data = df.to_dict(orient="records")

        # Aggregate quantities for each product
        for item in inventory_data:
            product = item["Product Name"]
            cutii = int(item["Cutii (Boxes)"]) if pd.notna(item["Cutii (Boxes)"]) else 0
            fractii = int(item["Fractii (Fractions)"]) if pd.notna(item["Fractii (Fractions)"]) else 0

            if product in existing_products:
                existing_products[product]["cutii"] += cutii
                existing_products[product]["fractii"] += fractii
            else:
                existing_products[product] = {"cutii": cutii, "fractii": fractii}

    return await render_template("inventory.html", location_name=location_name, inventory_data=inventory_data, existing_products=existing_products)

@app.route("/export_inventory/<location_name>")
async def export_inventory(location_name):
    today_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"inventories/{location_name}_inventory_{today_date}.xlsx"

    if await aiofiles.os.path.exists(filename):
        return await send_file(filename, as_attachment=True)

    return "Nu exista date de exportat", 404

@app.route("/delete_entry/<location_name>/<int:index>")
async def delete_entry(location_name, index):
    today_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"inventories/{location_name}_inventory_{today_date}.xlsx"

    if await aiofiles.os.path.exists(filename):
        df = await read_excel_async(filename) 

        # Ensure index is within valid range
        if 0 <= index < len(df):
            df = df.drop(index)
            await write_excel_async(df, filename) 

    return redirect(url_for("inventory_page", location_name=location_name))

@app.route("/search_products")
async def search_products():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify([])

    words = query.split()

    # Use an async session
    async with AsyncSessionFactory() as session:
        # Build an async query ensuring all words match
        stmt = select(Product).where(
            sa.and_(*(Product.name.ilike(f"%{word}%") for word in words))
        )
        result = await session.execute(stmt)
        products = result.scalars().all()  # Get results asynchronously

    matches = [{"product": p.name, "brand": p.brand} for p in products]
    return jsonify(matches)




###################################################### Main ######################################################
async def run():
    """Run app with Hypercorn."""
    await hypercorn.asyncio.serve(asgi_app, config)

# Entry point
if __name__ == "__main__":
    asyncio.run(run())