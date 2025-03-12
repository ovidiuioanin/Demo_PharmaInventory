# PharmaInventory - Pharmacy Inventory Management System

MedInventory is a web-based application designed to help pharmacies manage their inventory logging through technologies like OCR (Optical Character Recognition) and speech recognition. The system enables staff to quickly catalog products by scanning product labels or using voice input, streamlining the manual inventory logging process.

## Table of Contents

- [Technologies](#technologies)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Core Features](#core-features)
- [Documentation](#documentation)
- [Function Reference](#function-reference)

## Technologies

### Backend
- **Quart**: Asynchronous web framework compatible with Flask's API, chosen to handle concurrent requests efficiently
- **Hypercorn**: ASGI server that supports HTTP/2 and WebSockets, providing high-performance request handling
- **Socket.IO**: Real-time bidirectional event-based communication for OCR processing
- **SQLAlchemy**: ORM for database interaction with async support
- **aiosqlite**: Asynchronous SQLite interface
- **Pandas/Openpyxl**: For Excel file manipulation and data processing
- **aiofiles**: Asynchronous file operations

### Computer Vision & OCR
- **PaddleOCR**: OCR framework for text detection and recognition
- **OpenCV**: Image processing and manipulation
- **NumPy**: Numerical processing for image data

### Speech Recognition
- **Google Cloud Speech**: Advanced and easy-to-setup speech-to-text capabilities
- **Pydub**: Audio file manipulation and format conversion

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **JavaScript/jQuery**: Frontend interactivity and AJAX calls
- **Socket.IO Client**: Real-time communication with the server

### Utilities
- **Python-dotenv**: Environment variable management for configuration
- **Zipfile**: For creating archives of inventory data
- **UUID**: For generating unique identifiers needed for temporary OCR/STT files

## System Architecture

MedInventory uses an asynchronous architecture to handle multiple concurrent requests:

1. **Web Layer**: Quart provides the HTTP interface and routes
2. **Database Layer**: SQLAlchemy with aiosqlite for async database operations
3. **Processing Layer**:
   - OCR Processing: PaddleOCR for text detection in images
   - Speech Processing: Google Cloud Speech API for voice-to-text
4. **Communication Layer**: Socket.IO for real-time bidirectional communication
5. **Storage Layer**: File system for temporary files and Excel exports

The application leverages Python's asyncio capabilities throughout the stack to ensure non-blocking operations.


## Project Structure

```
MedInventory/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── data/
│   └── Database.xlsx     # Product database
├── static/
│   ├── styles.css        # CSS styles
│   └── img/              # Image assets
├── templates/
│   ├── admin.html        # Admin panel template
│   ├── admin_login.html  # Admin login template
│   ├── base.html         # Base template with common elements
│   ├── index.html        # Homepage template
│   ├── inventory.html    # Inventory management template
│   └── ocr.html          # OCR scanning page template
├── inventories/          # Storage for location-based inventory files
├── temp_audio/           # Temporary storage for audio files used for STT
└── debug/                # Debug image storage
```

## Core Features

### Inventory Management
- Track product stock across multiple pharmacy locations
- Add products with box and fraction counts
- Export inventory data to Excel
- View combined inventory statistics

### OCR Capability
- Scan product labels using device camera
- Process text from images using PaddleOCR
- Interactive selection of detected text
- Automatic product search using detected text

### Voice Input
- Record voice using device microphone
- Convert speech to text using Google Cloud Speech API
- Automatically fill in product search field

### Admin Functions
- Enable/disable inventory logging session
- Export all inventory data
- Reset inventory data
- Secure login for administrative access

## Documentation

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process_audio` | POST | Process audio recording and return transcribed text |
| `/ocr` | GET | Render OCR scanning page |
| `/admin` | GET/POST | Admin panel for system management |
| `/admin/login` | GET/POST | Admin authentication |
| `/admin/logout` | GET | Log out admin user |
| `/inventory` | GET | Main inventory selection page |
| `/inventory/<location_name>` | GET/POST | Location-specific inventory management |
| `/export_inventory/<location_name>` | GET | Export location inventory as Excel |
| `/delete_entry/<location_name>/<index>` | GET | Delete inventory entry |
| `/search_products` | GET | Search products API endpoint |

### Socket.IO Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | client→server | Client connection event |
| `disconnect` | client→server | Client disconnection event |
| `capture` | client→server | Send captured image for OCR processing |
| `processed_frame` | server→client | Return OCR processed image with text blocks |

## Function Reference

### Database Setup
- `init_database()`: Initializes the database and loads product data
- `get_db_session()`: Provides an async session for database operations

### Utility Functions
- `ensure_inventory_directory()`: Creates inventories directory if not exists
- `ensure_temp_audio_directory()`: Creates temporary audio directory if not exists
- `load_locations()`: Loads or creates the location configuration file
- `read_excel_async()`: Reads Excel files asynchronously
- `write_excel_async()`: Writes Excel files asynchronously
- `load_inventory_status()`: Loads inventory enable/disable status
- `save_inventory_status()`: Saves inventory enable/disable status
- `export_all_inventories()`: Creates a ZIP file with all inventory Excel files
- `reset_all_inventories()`: Deletes all inventory files

### OCR Functions
- `perform_ocr()`: Processes an image frame to extract text using PaddleOCR
- `handle_capture()`: Socket.IO handler for image capture and OCR processing

### Speech Recognition
- `convert_audio_to_wav()`: Converts audio files to WAV format for processing
- `transcribe_audio_google()`: Sends audio to Google Speech API for transcription

### Cleanup Functions
- `cleanup_startup()`: Removes debug files at application startup
- `setup_debug_cleanup()`: Sets up background task to clean up old debug files
- `setup_audio_cleanup()`: Sets up background task to clean up old audio files

### Web Routes
- Multiple route handlers for HTTP endpoints listed in the Documentation

