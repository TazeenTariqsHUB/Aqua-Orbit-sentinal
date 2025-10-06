from flask import Flask, render_template, Response, jsonify, request, send_from_directory, redirect, url_for
from flask_socketio import SocketIO
import cv2
import requests
from datetime import datetime
import threading
import time

# Import YOLOv5 API with error handling
try:
    from yolo_web_api import create_yolo_api
    YOLO_AVAILABLE = True
    print("YOLOv5 API imported successfully")
except ImportError as e:
    print(f"YOLOv5 API not available: {e}")
    print("Falling back to external detection API")
    YOLO_AVAILABLE = False
    create_yolo_api = None

app = Flask(__name__)
socketio = SocketIO(app)

# --------------------
# SQLite setup for tags
# --------------------
import sqlite3
from contextlib import closing

DB_PATH = 'tags.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with closing(get_db_connection()) as conn:
        cur = conn.cursor()
        cur.execute(
            '''CREATE TABLE IF NOT EXISTS tags (
                id TEXT NOT NULL,
                species TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                depth REAL,
                habitat TEXT,
                diet TEXT,
                water_temp_c REAL,
                salinity_psu REAL,
                pollution_index INTEGER,
                breeding_season INTEGER,
                environment_score INTEGER,
                timestamp TEXT NOT NULL
            )'''
        )
        cur.execute('CREATE INDEX IF NOT EXISTS idx_tags_id ON tags(id)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_tags_time ON tags(timestamp)')
        conn.commit()

init_db()
@app.route('/tigershark_images')
def tigershark_images():
    return render_template('tigershark_images.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

@app.route('/great_white_habitat')
def great_white_habitat():
    return render_template('great_white_habitat.html')

@app.route('/hammerhead_habitat')
def hammerhead_habitat():
    return render_template('Hammer_head.html')

@app.route('/Hammer_head')
def hammer_head():
    return render_template('Hammer_head.html')

@app.route('/hammer_head_images')
def hammer_head_images():
    return render_template('hammer_head_images.html')

@app.route('/report_generation')
def report_generation():
    return render_template('report_generation.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    # Handle both GET and POST requests
    if request.method == 'POST':
        query = request.form.get('location', '').lower()
    else:
        query = request.args.get('q', '').lower()
    
    # Check if the search query is related to satellite shark tracking
    shark_keywords = ['satellite', 'shark', 'track', 'migration', 'orbit', 'ocean']
    if any(keyword in query for keyword in shark_keywords):
        return redirect(url_for('article1'))
    
    # For other searches, redirect back to home with a message
    return redirect(url_for('index'))
# Initialize camera lazily
cap = None
camera_lock = threading.Lock()

# Share latest frame safely between video stream and detection endpoint
latest_frame = None
latest_frame_lock = threading.Lock()

# Latest detections and the frame size used when detecting
last_predictions = []
last_pred_size = (0, 0)  # (width, height)
predictions_lock = threading.Lock()

latest_top = {
    'species': None,
    'confidence': 0.0,
    'time': None
}

species_info = {
    'Sand Tiger Shark': {
        'habitat': ['Coastal waters', 'Sandy bottoms', 'Rocky reefs'],
        'characteristics': ['Stout body', 'Pointed snout', 'Protruding teeth', 'Air gulping'],
        'feed': ['Bony fish', 'Smaller sharks', 'Crustaceans']
    },
    'sand tiger shark': {
        'habitat': ['Coastal waters', 'Sandy bottoms', 'Rocky reefs'],
        'characteristics': ['Stout body', 'Pointed snout', 'Protruding teeth', 'Air gulping'],
        'feed': ['Bony fish', 'Smaller sharks', 'Crustaceans']
    },
    'Sand_Tiger_Shark': {
        'habitat': ['Coastal waters', 'Sandy bottoms', 'Rocky reefs'],
        'characteristics': ['Stout body', 'Pointed snout', 'Protruding teeth', 'Air gulping'],
        'feed': ['Bony fish', 'Smaller sharks', 'Crustaceans']
    },
    'Carcharias taurus': {
        'habitat': ['Coastal waters', 'Sandy bottoms', 'Rocky reefs'],
        'characteristics': ['Stout body', 'Pointed snout', 'Protruding teeth', 'Air gulping'],
        'feed': ['Bony fish', 'Smaller sharks', 'Crustaceans']
    },
    'Sandbar Shark': {
        'habitat': ['Coastal waters', 'Sandy bottoms', 'Rocky reefs'],
        'characteristics': ['Stout body', 'Pointed snout'],
        'feed': ['Bony fish', 'Crustaceans']
    },
    'Great White Shark': {
        'habitat': ['Coastal and open oceans', 'Temperate waters'],
        'characteristics': ['Large size', 'Powerful jaws', 'Streamlined body'],
        'feed': ['Seals', 'fish', 'smaller sharks']
    },
    'Tiger Shark': {
        'habitat': ['Tropical and subtropical waters'],
        'characteristics': ['Striped pattern', 'Aggressive predator'],
        'feed': ['Fish', 'turtles', 'birds', 'mammals']
    },
    'Bull Shark': {
        'habitat': ['Coastal waters', 'rivers', 'estuaries'],
        'characteristics': ['Aggressive', 'adaptable to freshwater'],
        'feed': ['Fish', 'dolphins', 'other sharks']
    },
    'shark': {
        'habitat': ['Varies by species'],
        'characteristics': ['Cartilaginous fish', 'Predatory'],
        'feed': ['Varies by species']
    },
    'unknown shark': {
        'habitat': ['Unknown'],
        'characteristics': ['Cartilaginous fish', 'Predatory'],
        'feed': ['Unknown']
    }
}

def generate_frames():
    print("Starting video feed generation...")
    # Try to open camera if not already open
    if cap is None or not cap.isOpened():
        print("Camera not open, attempting to open laptop camera...")
        if not open_camera():
            print("Failed to open laptop camera, showing placeholder")
            # Return a placeholder frame if camera is not available
            import numpy as np
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Laptop Camera Not Available", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(placeholder, "Check if camera is connected", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(placeholder, "and not in use by another app", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            encoded = cv2.imencode('.jpg', placeholder)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')
            return
    
    print("Camera is open, starting video stream...")
    
    frame_count = 0
    while True:
        with camera_lock:
            is_open = cap is not None and cap.isOpened()
        if not is_open:
            print("Camera not open in generate_frames loop")
            break
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera")
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Log every 30 frames (about once per second)
            print(f"Generated {frame_count} frames from camera")
        # Update latest_frame for detection endpoint
        with latest_frame_lock:
            global latest_frame
            latest_frame = frame.copy()
        # Draw latest detections onto the frame
        with predictions_lock:
            preds = list(last_predictions)
            det_w, det_h = last_pred_size
        if preds and det_w > 0 and det_h > 0:
            h, w = frame.shape[:2]
            scale_x = float(w) / float(det_w)
            scale_y = float(h) / float(det_h)
            for p in preds:
                try:
                    cx = float(p.get('x', 0.0)) * scale_x
                    cy = float(p.get('y', 0.0)) * scale_y
                    bw = float(p.get('width', 0.0)) * scale_x
                    bh = float(p.get('height', 0.0)) * scale_y
                    x1 = max(0, int(cx - bw / 2))
                    y1 = max(0, int(cy - bh / 2))
                    x2 = min(w - 1, int(cx + bw / 2))
                    y2 = min(h - 1, int(cy + bh / 2))
                    label = p.get('class') or p.get('label') or 'object'
                    conf = float(p.get('confidence', 0.0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 2)
                    text = f"{label} {int(conf * 100)}%"
                    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    rx1, ry1 = x1, max(0, y1 - th - baseline - 4)
                    rx2, ry2 = x1 + tw + 6, y1
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 215, 255), thickness=-1)
                    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                except Exception:
                    continue
        # Always yield a frame, even if there are no detections
        encoded = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/shark-tracking')
def shark_tracking():
    return render_template('shark_tracking.html')

@app.route('/shark_tracking')
def shark_tracking_alt():
    return render_template('shark_tracking.html')


@app.route('/3d-ocean-view')
def ocean_view():  # or _3d_ocean_view
    return render_template('3D_ocean_veiw.html')

@app.route('/article1')
def article1():
    return render_template('article1.html')

@app.route('/article2')
def article2():
    return render_template('article2.html')

@app.route('/article3')
def article3():
    return render_template('article3.html')

@app.route('/article4')
def article4():
    return render_template('article4.html')

@app.route('/article5')
def article5():
    return render_template('article5.html')


@app.route('/video_feed')
def video_feed():
    print("Video feed requested for laptop camera...")
    try:
        # Ensure camera is open before starting stream
        if cap is None or not cap.isOpened():
            print("Opening laptop camera for video feed...")
            if not open_camera():
                print("Failed to open laptop camera for video feed")
                # Return error frame
                import numpy as np
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Laptop Camera Error", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(error_frame, "Failed to open camera", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(error_frame, "Check camera connection", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                encoded = cv2.imencode('.jpg', error_frame)[1].tobytes()
                return Response(
                    b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n',
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                )
        
        print("Laptop camera is open, starting video stream...")
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {e}")
        # Return a simple error response
        import numpy as np
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Laptop Camera Error: {str(e)}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        encoded = cv2.imencode('.jpg', error_frame)[1].tobytes()
        return Response(
            b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n',
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )


@app.route('/detect_fish', methods=['GET'])
def detect_fish():
    # Use the most recent frame from the live stream to avoid camera contention
    with latest_frame_lock:
        frame = None if (globals().get('latest_frame') is None) else latest_frame.copy()

    if frame is None:
        return jsonify({
            'error': 'No frame available yet. Open the live stream or wait a moment.'
        }), 503

    # Resize to a robust working width and enhance contrast for printed images
    h, w = frame.shape[:2]
    target_w = 1024
    if w > target_w:
        scale = target_w / float(w)
        new_size = (int(w * scale), int(h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # Enhanced contrast enhancement (CLAHE with higher clip limit)
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clipLimit for better contrast
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        frame = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    except Exception:
        pass

    success, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return jsonify({
            'error': 'Failed to encode image'
        }), 500

    image_bytes = encoded_image.tobytes()

    # Updated to shark-specific model
    url = 'https://detect.roboflow.com/sharkdetection-ykdcy/1'
    params = {
        'api_key': 'YregzS3X1k22xaPX0IMS',  # Your API key
        'confidence': '0.2',  # Further lowered threshold
        'overlap': '0.3'
    }
    files = {
        'file': ('frame.jpg', image_bytes, 'image/jpeg')
    }

    try:
        response = requests.post(url, params=params, files=files, timeout=20)
        response.raise_for_status()
        data = response.json()
        try:
            preds_dbg = data.get('predictions') or data.get('detections') or []
            if preds_dbg:
                top_dbg = max(preds_dbg, key=lambda p: p.get('confidence', 0))
                print('[detect_fish] top:', top_dbg.get('class'), top_dbg.get('confidence'))
            else:
                print('[detect_fish] no predictions')
        except Exception:
            pass
    except Exception as e:
        return jsonify({
            'error': 'Detection request failed',
            'details': str(e)
        }), 502

    predictions = data.get('predictions') or data.get('detections') or []
    # Handle classification-style response (dict of class -> confidence)
    if isinstance(predictions, dict):
        if not predictions:
            return jsonify({
                'species': None,
                'confidence': 0.0,
                'message': 'No shark detected',
                'raw': data
            })
        top_label, top_info = max(predictions.items(), key=lambda kv: kv[1].get('confidence', 0))
        species = top_label
        confidence = float(top_info.get('confidence', 0))
        info = species_info.get(species) or species_info.get(species.replace('-', ' ')) or species_info.get(
            species.replace('_', ' ')) or species_info.get(species.lower()) or species_info.get('shark')
        return jsonify({
            'species': species,
            'confidence': confidence,
            'time': datetime.now().strftime('%H:%M:%S'),
            'info': info
        })

    if not predictions:
        return jsonify({
            'species': None,
            'confidence': 0.0,
            'message': 'No shark detected',
            'raw': data
        })

    top = max(predictions, key=lambda p: p.get('confidence', 0))
    species = top.get('class') or top.get('label') or 'unknown shark'
    confidence = float(top.get('confidence', 0))

    info = species_info.get(species) or species_info.get(species.replace('-', ' ')) or species_info.get(
        species.replace('_', ' ')) or species_info.get(species.lower()) or species_info.get('shark')
    return jsonify({
        'species': species,
        'confidence': confidence,
        'time': datetime.now().strftime('%H:%M:%S'),
        'info': info
    })


def _detection_worker(stop_event: threading.Event, interval_seconds: float = 2.6, detect_width: int = 1024):
    while not stop_event.is_set():
        frame_copy = None
        with latest_frame_lock:
            if globals().get('latest_frame') is not None:
                frame_copy = latest_frame.copy()

        if frame_copy is not None:
            # Resize for detection
            h, w = frame_copy.shape[:2]
            if w > detect_width:
                scale = detect_width / float(w)
                new_size = (int(w * scale), int(h * scale))
                frame_copy = cv2.resize(frame_copy, new_size, interpolation=cv2.INTER_AREA)

            # Enhanced contrast enhancement
            try:
                lab = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l2 = clahe.apply(l)
                lab2 = cv2.merge((l2, a, b))
                frame_copy = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
            except Exception:
                pass

            ok, enc = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                image_bytes = enc.tobytes()
                # Updated to shark-specific model
                url = 'https://detect.roboflow.com/sharkdetection-ykdcy/1'
                params = {'api_key': 'YregzS3X1k22xaPX0IMS', 'confidence': '0.2', 'overlap': '0.3'}
                files = {'file': ('frame.jpg', image_bytes, 'image/jpeg')}
                try:
                    resp = requests.post(url, params=params, files=files, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    preds = data.get('predictions') or data.get('detections') or []
                    with predictions_lock:
                        global last_predictions, last_pred_size
                        hh, ww = frame_copy.shape[:2]
                        last_pred_size = (ww, hh)
                        if isinstance(preds, dict):
                            last_predictions = []  # no boxes to draw
                            if preds:
                                top_label, top_info = max(preds.items(), key=lambda kv: kv[1].get('confidence', 0))
                                latest_top['species'] = top_label
                                latest_top['confidence'] = float(top_info.get('confidence', 0))
                                latest_top['time'] = datetime.now().strftime('%H:%M:%S')
                        else:
                            last_predictions = preds
                            if preds:
                                top = max(preds, key=lambda p: p.get('confidence', 0))
                                latest_top['species'] = top.get('class') or top.get('label') or 'unknown shark'
                                latest_top['confidence'] = float(top.get('confidence', 0))
                                latest_top['time'] = datetime.now().strftime('%H:%M:%S')
                except Exception:
                    # Ignore this cycle on error
                    pass

        stop_event.wait(interval_seconds)


# Detection thread handles (lazy start)
_detector_stop = None
_detector_thread = None


def open_camera() -> bool:
    global cap
    with camera_lock:
        if cap is not None and getattr(cap, 'isOpened', lambda: False)():
            print("Camera already open")
            return True
        
        try:
            print("Attempting to open laptop camera...")
            # Try camera index 0 first (usually laptop camera)
            cap = cv2.VideoCapture(0)
            if cap is not None and cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print("Laptop camera opened successfully on index 0")
                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for real-time feed
                    print(f"Camera properties set - Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                    return True
                else:
                    print("Camera opened but cannot read frames")
                    cap.release()
                    cap = None
            
            print("Failed to open camera on index 0 - trying alternative indices")
            # Try alternative camera indices
            for i in range(1, 5):  # Try cameras 1-4
                print(f"Trying camera index {i}...")
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    # Test if we can read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"Camera opened successfully on index {i}")
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        return True
                    else:
                        print(f"Camera {i} opened but cannot read frames")
                if cap is not None:
                    cap.release()
                cap = None
            
            print("No working camera found on any index")
            return False
        except Exception as e:
            print(f"Error opening camera: {e}")
            return False


def close_camera():
    global cap
    with camera_lock:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cap = None
    # Clear shared state
    with latest_frame_lock:
        global latest_frame
        latest_frame = None
    with predictions_lock:
        global last_predictions, last_pred_size
        last_predictions = []
        last_pred_size = (0, 0)


def start_detection_thread():
    global _detector_stop, _detector_thread
    if _detector_thread is not None and _detector_thread.is_alive():
        return
    _detector_stop = threading.Event()
    _detector_thread = threading.Thread(target=_detection_worker, args=(_detector_stop,), daemon=True)
    _detector_thread.start()


def stop_detection_thread():
    global _detector_stop, _detector_thread
    if _detector_stop is not None:
        _detector_stop.set()
    _detector_thread = None


@app.route('/latest_detection', methods=['GET'])
def latest_detection():
    with predictions_lock:
        return jsonify(latest_top)


@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    try:
        if not open_camera():
            return jsonify({'ok': False, 'error': 'Camera failed to open'}), 500
        start_detection_thread()
        return jsonify({'ok': True, 'message': 'Camera started successfully'})
    except Exception as e:
        print(f"Error starting camera: {e}")
        return jsonify({'ok': False, 'error': f'Camera error: {str(e)}'}), 500


@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    stop_detection_thread()
    close_camera()
    return jsonify({'ok': True})

@app.route('/camera_status', methods=['GET'])
def camera_status():
    """Check if camera is available and working"""
    try:
        with camera_lock:
            is_open = cap is not None and cap.isOpened()
        return jsonify({
            'ok': True,
            'camera_available': is_open,
            'message': 'Camera is working' if is_open else 'Camera not initialized'
        })
    except Exception as e:
        return jsonify({
            'ok': False,
            'camera_available': False,
            'error': str(e)
        }), 500

@app.route('/initialize_camera', methods=['POST'])
def initialize_camera():
    """Initialize camera for picture recognition"""
    try:
        print("Initializing camera...")
        
        # Force close any existing camera first
        close_camera()
        
        # Try to open camera
        if not open_camera():
            print("Failed to open camera")
            return jsonify({'ok': False, 'error': 'Failed to open camera - check if camera is connected and not in use by another application'}), 500
        
        print("Camera opened successfully")
        
        # Start detection thread for real-time processing
        start_detection_thread()
        
        return jsonify({
            'ok': True,
            'message': 'Camera initialized successfully',
            'video_feed_url': '/video_feed',
            'camera_index': 'Camera opened on available index'
        })
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return jsonify({'ok': False, 'error': f'Camera initialization failed: {str(e)}'}), 500

@app.route('/force_camera_start', methods=['POST'])
def force_camera_start():
    """Force start laptop camera and return status"""
    try:
        print("Force starting laptop camera...")
        
        # Close any existing camera
        close_camera()
        
        # Try multiple camera indices
        success = False
        camera_index = -1
        
        for i in range(5):  # Try cameras 0-4
            print(f"Trying laptop camera index {i}...")
            cap_temp = cv2.VideoCapture(i)
            if cap_temp is not None and cap_temp.isOpened():
                # Test if we can read a frame
                ret, frame = cap_temp.read()
                if ret and frame is not None:
                    print(f"Laptop camera {i} is working! Frame size: {frame.shape}")
                    cap_temp.release()
                    # Now open it properly
                    global cap
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        success = True
                        camera_index = i
                        print(f"Laptop camera {i} configured successfully")
                        break
                cap_temp.release()
        
        if success:
            start_detection_thread()
            return jsonify({
                'ok': True,
                'message': f'Laptop camera started successfully on index {camera_index}',
                'camera_index': camera_index,
                'video_feed_url': '/video_feed'
            })
        else:
            return jsonify({
                'ok': False,
                'error': 'No working laptop camera found on any index (0-4)'
            }), 500
            
    except Exception as e:
        print(f"Error force starting laptop camera: {e}")
        return jsonify({'ok': False, 'error': f'Laptop camera force start failed: {str(e)}'}), 500

@app.route('/test_camera', methods=['GET'])
def test_camera():
    """Test if laptop camera is working"""
    try:
        print("Testing laptop camera...")
        cap_test = cv2.VideoCapture(0)
        if cap_test is not None and cap_test.isOpened():
            ret, frame = cap_test.read()
            if ret and frame is not None:
                cap_test.release()
                return jsonify({
                    'ok': True,
                    'message': 'Laptop camera is working',
                    'frame_size': f"{frame.shape[1]}x{frame.shape[0]}"
                })
            else:
                cap_test.release()
                return jsonify({
                    'ok': False,
                    'error': 'Camera opened but cannot read frames'
                }), 500
        else:
            return jsonify({
                'ok': False,
                'error': 'Cannot open laptop camera'
            }), 500
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': f'Camera test failed: {str(e)}'
        }), 500


@app.route('/plotly_globe')
def plotly_globe():
    return render_template('plotly_globe.html')


@app.route('/realtime-tracking')
def realtime_tracking():
    """Render the real-time shark tracking page"""
    return render_template('realtime_tracking.html')

@app.route('/greatwhite_habitat')
def greatwhite_habitat():
    """Render the Great White Shark habitat page with 3D model"""
    return render_template('greatwhite_habitat.html')


# Optional: Bootstrap DB with fish_tags.json once if DB empty
import json
import os

def bootstrap_from_json():
    try:
        with closing(get_db_connection()) as conn:
            cur = conn.cursor()
            cur.execute('SELECT COUNT(1) FROM tags')
            count = cur.fetchone()[0]
            if count:
                return
            if not os.path.exists('fish_tags.json'):
                return
            with open('fish_tags.json', 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                return
            for d in data:
                cur.execute('''INSERT INTO tags (
                    id, species, lat, lon, depth, habitat, diet, water_temp_c, salinity_psu,
                    pollution_index, breeding_season, environment_score, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    d.get('id'), d.get('species'), float(d.get('lat')), float(d.get('lon')),
                    d.get('depth'), d.get('habitat'), d.get('diet'), d.get('water_temp_c'),
                    d.get('salinity_psu'), d.get('pollution_index'),
                    1 if d.get('breeding_season') else 0 if d.get('breeding_season') is not None else None,
                    d.get('environment_score'), d.get('timestamp')
                ))
            conn.commit()
    except Exception:
        pass

bootstrap_from_json()


@app.route('/tag-data', methods=['POST'])
def receive_tag_data():
    """Receive tag telemetry and store in SQLite"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    if 'timestamp' not in data:
        data['timestamp'] = datetime.utcnow().isoformat() + 'Z'

    required = ['id', 'species', 'lat', 'lon']
    for k in required:
        if data.get(k) is None:
            return jsonify({"error": f"Missing required field: {k}"}), 400

    try:
        with closing(get_db_connection()) as conn:
            cur = conn.cursor()
            cur.execute('''INSERT INTO tags (
                id, species, lat, lon, depth, habitat, diet, water_temp_c, salinity_psu,
                pollution_index, breeding_season, environment_score, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                data.get('id'), data.get('species'), float(data.get('lat')), float(data.get('lon')),
                data.get('depth'), data.get('habitat'), data.get('diet'), data.get('water_temp_c'),
                data.get('salinity_psu'), data.get('pollution_index'),
                1 if data.get('breeding_season') else 0 if data.get('breeding_season') is not None else None,
                data.get('environment_score'), data.get('timestamp')
            ))
            conn.commit()
    except Exception as e:
        return jsonify({"error": "Failed to save", "details": str(e)}), 500

    return jsonify({"status": "success", "received": data}), 200


@app.route('/get-tags', methods=['GET'])
def get_tags():
    """Retrieve all tag data from SQLite, newest first per id"""
    try:
        with closing(get_db_connection()) as conn:
            cur = conn.cursor()
            # Return most recent entry per id; fallback to all if SQLite version lacks window functions
            try:
                cur.execute('''
                    SELECT id, species, lat, lon, depth, habitat, diet, water_temp_c, salinity_psu,
                           pollution_index, breeding_season, environment_score, timestamp
                    FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY timestamp DESC) AS rn
                        FROM tags
                    ) t
                    WHERE rn = 1
                    ORDER BY timestamp DESC
                ''')
                rows = cur.fetchall()
            except Exception:
                cur.execute('SELECT id, species, lat, lon, depth, habitat, diet, water_temp_c, salinity_psu, pollution_index, breeding_season, environment_score, timestamp FROM tags ORDER BY timestamp DESC')
                rows = cur.fetchall()
            result = []
            for r in rows:
                item = {k: r[k] for k in r.keys()}
                if item.get('breeding_season') is not None:
                    item['breeding_season'] = bool(item['breeding_season'])
                result.append(item)
            return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Failed to load tags", "details": str(e)}), 500


# Initialize YOLOv5 API endpoints if available
if YOLO_AVAILABLE and create_yolo_api:
    try:
        create_yolo_api(app)
        print("YOLOv5 API endpoints initialized successfully")
    except Exception as e:
        print(f"Failed to initialize YOLOv5 API: {e}")
        YOLO_AVAILABLE = False
else:
    print("YOLOv5 API not available - using external detection services")
    
    # Add fallback API endpoints
    @app.route('/api/yolo/status', methods=['GET'])
    def yolo_status_fallback():
        return jsonify({
            'success': False,
            'model_loaded': False,
            'error': 'YOLOv5 model not available - using external detection'
        })
    
    @app.route('/api/yolo/detect', methods=['POST'])
    def yolo_detect_fallback():
        return jsonify({
            'success': False,
            'error': 'YOLOv5 model not available - using external detection services',
            'fallback': 'Use /detect_fish endpoint instead'
        })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)