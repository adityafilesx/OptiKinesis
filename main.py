import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import math
import threading
from flask import Flask, render_template, Response, request, jsonify
import webbrowser
import time
import urllib.parse
import sys
import os
from collections import deque

# --- PyQt5 IMPORTS FOR CURSOR OVERLAY ---
try:
    from PyQt5 import QtWidgets, QtGui, QtCore
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    print("WARNING: PyQt5 not installed. Cursor overlay will not be available.")
    print("Install with: pip install PyQt5")

# --- TWILIO IMPORT (Optional) ---
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# --- FATIGUE MONITORING SYSTEM ---
from fatigue_monitor import fatigue_monitor

# --- VOICE COMMAND AUTOMATION ---
from voice_commands import voice_executor

app = Flask(__name__)

# --- CONFIGURATION ---
pyautogui.FAILSAFE = False

# Screen settings
SCREEN_W, SCREEN_H = pyautogui.size()
CENTER_X = SCREEN_W // 2
CENTER_Y = SCREEN_H // 2

# Head-pose tracking settings (accessible speed)
FILTER_LENGTH = 12  # More frames = smoother cursor
YAW_DEGREES = 25    # Degrees left/right to reach screen edge
PITCH_DEGREES = 18  # Degrees up/down to reach screen edge

# Axis inversion flags (correct camera mirroring)
# Set True to invert axis: look LEFT -> cursor LEFT, look RIGHT -> cursor RIGHT
INVERT_X = True    # Invert horizontal (for front-facing camera mirror correction)
INVERT_Y = False   # Invert vertical (usually not needed)

# Blink Settings (tuned for reliable 2-blink click)
BLINK_THRESH = 0.2
BLINK_DURATION_MIN = 0.15  # Minimum blink hold time (150ms)
BLINK_COOLDOWN = 0.25      # Time between blinks
BLINKS_TO_CLICK = 2        # Require 2 blinks to trigger click
BLINK_WINDOW = 1.5         # 1.5 second window to complete 2 blinks

# Face landmark indices for bounding box (from MonitorTracking.py)
LANDMARKS = {
    "left": 234,
    "right": 454,
    "top": 10,
    "bottom": 152,
    "front": 1,
}

# Configurable Settings
SETTINGS = {
    "emergency_contact": "+916387533207",
    "cursor_scope": 1.0,
    "blink_sensitivity": BLINK_THRESH,
    "cursor_speed": 0.15,  # Ultra slow for users with severe medical conditions
    "overlay_enabled": True,
    "mouse_control_enabled": True
}

# EMA smoothing for cursor position
prev_screen_x = CENTER_X
prev_screen_y = CENTER_Y

# Calibration offsets
calibration_offset_yaw = 0
calibration_offset_pitch = 0
calibration_lock = threading.Lock()

# Ray smoothing buffers
ray_origins = deque(maxlen=FILTER_LENGTH)
ray_directions = deque(maxlen=FILTER_LENGTH)

# Shared mouse target position (thread-safe)
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

# --- TWILIO CREDENTIALS (Optional) ---
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")

# --- CURSOR CIRCLE OVERLAY (from CursorCircle.py) ---
if PYQT5_AVAILABLE:
    class CursorOverlay(QtWidgets.QWidget):
        def __init__(self, radius=80):
            super().__init__()
            self.radius = radius
            self.diameter = 2 * self.radius + 4
            self.setWindowFlags(
                QtCore.Qt.FramelessWindowHint |
                QtCore.Qt.WindowStaysOnTopHint |
                QtCore.Qt.Tool |
                QtCore.Qt.X11BypassWindowManagerHint
            )
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
            self.setFixedSize(self.diameter, self.diameter)

            self.label = QtWidgets.QLabel(self)
            self.label.setGeometry(0, 0, self.diameter, self.diameter)

            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_position)
            self.timer.start(10)

        def update_position(self):
            if SETTINGS.get("overlay_enabled", True):
                x, y = pyautogui.position()
                self.move(x - self.radius, y - self.radius)
                self.draw_circle()
                self.show()
            else:
                self.hide()

        def draw_circle(self):
            img = np.zeros((self.diameter, self.diameter, 4), dtype=np.uint8)
            cv2.circle(img, (self.radius + 2, self.radius + 2), self.radius - 5, (0, 255, 0, 255), 10)
            qimg = QtGui.QImage(img.data, self.diameter, self.diameter, QtGui.QImage.Format_RGBA8888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.label.setPixmap(pixmap)


def run_overlay():
    """Run the cursor overlay in a separate thread with its own QApplication."""
    if not PYQT5_AVAILABLE:
        return
    qt_app = QtWidgets.QApplication([])
    overlay = CursorOverlay(radius=80)
    overlay.show()
    qt_app.exec_()


# --- MOUSE MOVER THREAD ---
def mouse_mover():
    """Continuously move mouse to target position (smoother than direct control in main loop)."""
    while True:
        if SETTINGS.get("mouse_control_enabled", True):
            with mouse_lock:
                x, y = mouse_target
            try:
                if not (math.isnan(x) or math.isnan(y)):
                    pyautogui.moveTo(x, y)
            except Exception:
                pass
        time.sleep(0.01)


# --- HELPERS ---
def landmark_to_np(landmark, w, h):
    """Convert MediaPipe landmark to numpy array with pixel coordinates."""
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])


def make_twilio_call():
    if not TWILIO_AVAILABLE:
        return
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        client.calls.create(
            twiml='<Response><Say>Emergency alert. User triggered SOS.</Say></Response>',
            to=SETTINGS["emergency_contact"],
            from_=TWILIO_PHONE
        )
    except:
        pass


def auto_send_whatsapp(number, message):
    msg_encoded = urllib.parse.quote(message)
    link = f"https://web.whatsapp.com/send?phone={number}&text={msg_encoded}"
    webbrowser.open(link)
    time.sleep(20)
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.press('enter')


def execute_type_external(text):
    time.sleep(5)
    pyautogui.write(text, interval=0.1)


def get_blink_ratio(landmarks, eye_indices):
    """Calculate eye aspect ratio for blink detection."""
    top = landmarks[eye_indices[1]]
    bottom = landmarks[eye_indices[3]]
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[2]]
    
    ver_dist = math.hypot(top.x - bottom.x, top.y - bottom.y)
    hor_dist = math.hypot(left.x - right.x, left.y - right.y)
    
    return ver_dist / hor_dist if hor_dist != 0 else 0


# --- MEDIAPIPE SETUP (Tasks API for 0.10.30+) ---
# Download the face landmarker model if not present
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"Downloading face landmarker model to {MODEL_PATH}...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete!")

# Create Face Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

print("Initializing Camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera 0 could not be opened. Trying 1...")
    cap = cv2.VideoCapture(1)


def gen_frames():
    """Generate video frames with head-pose tracking (from MonitorTracking.py)."""
    global calibration_offset_yaw, calibration_offset_pitch
    
    # Blink State for 3-blink click
    blink_start_time = 0
    last_click_time = 0
    blink_active = False
    blink_count = 0          # Count blinks in window
    blink_timestamps = []    # Track blink times for 3-blink detection
    LEFT_EYE = [33, 159, 133, 145]

    print("Camera Loop Started with Head-Pose Tracking.")

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        h, w, _ = frame.shape
        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format for Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = face_landmarker.detect(mp_image)

        if results.face_landmarks and len(results.face_landmarks) > 0:
            face_landmarks = results.face_landmarks[0]

            # --- HEAD-POSE TRACKING (from MonitorTracking.py) ---
            # Extract key points for orientation
            key_points = {}
            for name, idx in LANDMARKS.items():
                pt = landmark_to_np(face_landmarks[idx], w, h)
                key_points[name] = pt
                x, y = int(pt[0]), int(pt[1])
                
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            left = key_points["left"]
            right = key_points["right"]
            top = key_points["top"]
            bottom = key_points["bottom"]
            front = key_points["front"]

            # Compute oriented axes based on head geometry
            right_axis = (right - left)
            right_axis /= np.linalg.norm(right_axis)

            up_axis = (top - bottom)
            up_axis /= np.linalg.norm(up_axis)

            forward_axis = np.cross(right_axis, up_axis)
            forward_axis /= np.linalg.norm(forward_axis)
            forward_axis = -forward_axis  # Flip to face outward

            # Compute center of the head
            center = (left + right + top + bottom + front) / 5

            # Update smoothing buffers
            ray_origins.append(center)
            ray_directions.append(forward_axis)

            # Compute averaged ray direction
            avg_origin = np.mean(ray_origins, axis=0)
            avg_direction = np.mean(ray_directions, axis=0)
            avg_direction /= np.linalg.norm(avg_direction)

            # Reference forward direction
            reference_forward = np.array([0, 0, -1])

            # Horizontal (yaw) angle
            xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
            xz_proj /= np.linalg.norm(xz_proj)
            yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
            if avg_direction[0] < 0:
                yaw_rad = -yaw_rad

            # Vertical (pitch) angle
            yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
            yz_proj /= np.linalg.norm(yz_proj)
            pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
            if avg_direction[1] > 0:
                pitch_rad = -pitch_rad

            # Convert to degrees
            yaw_deg = np.degrees(yaw_rad)
            pitch_deg = np.degrees(pitch_rad)

            # Normalize angles (from MonitorTracking.py)
            if yaw_deg < 0:
                yaw_deg = abs(yaw_deg)
            elif yaw_deg < 180:
                yaw_deg = 360 - yaw_deg

            if pitch_deg < 0:
                pitch_deg = 360 + pitch_deg

            raw_yaw_deg = yaw_deg
            raw_pitch_deg = pitch_deg

            # Apply calibration offsets
            with calibration_lock:
                yaw_deg += calibration_offset_yaw
                pitch_deg += calibration_offset_pitch

            # Map to screen coordinates
            screen_x = int(((yaw_deg - (180 - YAW_DEGREES)) / (2 * YAW_DEGREES)) * SCREEN_W)
            screen_y = int(((180 + PITCH_DEGREES - pitch_deg) / (2 * PITCH_DEGREES)) * SCREEN_H)

            # Apply axis inversion to correct camera mirroring
            # INVERT_X: flip horizontal so look LEFT = cursor LEFT
            # INVERT_Y: flip vertical so look UP = cursor UP
            if INVERT_X:
                screen_x = SCREEN_W - screen_x
            if INVERT_Y:
                screen_y = SCREEN_H - screen_y

            # Clamp to screen bounds
            screen_x = max(10, min(SCREEN_W - 10, screen_x))
            screen_y = max(10, min(SCREEN_H - 10, screen_y))

            # Apply EMA smoothing for accessible speed
            global prev_screen_x, prev_screen_y
            ema_alpha = 0.15  # Low alpha = slower, smoother cursor
            screen_x = int(prev_screen_x + ema_alpha * (screen_x - prev_screen_x))
            screen_y = int(prev_screen_y + ema_alpha * (screen_y - prev_screen_y))
            prev_screen_x = screen_x
            prev_screen_y = screen_y

            # Update mouse target
            if SETTINGS.get("mouse_control_enabled", True):
                with mouse_lock:
                    mouse_target[0] = screen_x
                    mouse_target[1] = screen_y

            # Draw gaze direction ray on frame
            half_depth = 80
            ray_length = 2.5 * half_depth
            ray_end = avg_origin - avg_direction * ray_length
            cv2.line(frame, (int(avg_origin[0]), int(avg_origin[1])), 
                    (int(ray_end[0]), int(ray_end[1])), (15, 255, 0), 3)

            # --- BLINK DETECTION with FATIGUE MONITORING ---
            ratio = get_blink_ratio(face_landmarks, LEFT_EYE)
            current_time = time.time()
            
            # Use fatigue-adjusted sensitivity (reduced during fatigue)
            base_thresh = SETTINGS["blink_sensitivity"]
            thresh = fatigue_monitor.get_adjusted_sensitivity(base_thresh)

            if ratio < thresh:
                blink_active = True
            else:
                if blink_active:
                    # Blink completed - record for fatigue monitoring
                    fatigue_triggered = fatigue_monitor.record_blink()
                    
                    try:
                        pyautogui.click()
                        blink_active = False
                    except:
                        pass
            
            # Get fatigue status for UI display
            fatigue_status = fatigue_monitor.check_fatigue()
            #     # Eye is closed
            #     if blink_start_time == 0:
            #         blink_start_time = current_time

            #     blink_duration = current_time - blink_start_time

            #     # Register blink after minimum duration (and not already registered)
            #     if blink_duration > BLINK_DURATION_MIN and not blink_triggered:
            #         blink_triggered = True
            #         blink_timestamps.append(current_time)
                    
            #         # Remove old blinks outside the time window
            #         blink_timestamps = [t for t in blink_timestamps if current_time - t < BLINK_WINDOW]
                    
            #         # Check if we have 2 blinks within the window
            #         if len(blink_timestamps) >= BLINKS_TO_CLICK:
            #             if (current_time - last_click_time) > BLINK_COOLDOWN:
            #                 try:
            #                     pyautogui.click()
            #                     # Play click sound (macOS)
            #                     import subprocess
            #                     subprocess.Popen(['afplay', '/System/Library/Sounds/Tink.aiff'], 
            #                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #                     last_click_time = current_time
            #                     blink_timestamps = []  # Reset after click
            #                 except:
            #                     pass

            #     cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)
            # else:
            #     blink_start_time = 0
            #     blink_triggered = False

            # Remove expired blinks from window
            blink_timestamps = [t for t in blink_timestamps if current_time - t < BLINK_WINDOW]
            
            # Show blink count feedback
            blink_count_display = len(blink_timestamps)
            cv2.putText(frame, f"Blinks: {blink_count_display}/2", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display tracking info
            cv2.putText(frame, f"Yaw: {raw_yaw_deg:.1f} Pitch: {raw_pitch_deg:.1f}", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Blink 2x to click | 'c' to calibrate", 
                       (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        else:
            cv2.putText(frame, "FACE NOT DETECTED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/calibrate', methods=['POST'])
def calibrate():
    """Calibrate the gaze tracking to center on current head position."""
    global calibration_offset_yaw, calibration_offset_pitch
    
    # Get current raw values from the tracking loop
    # For now, we set a flag that will be read in the next frame
    with calibration_lock:
        # Reset offsets - the next frame will recalibrate
        calibration_offset_yaw = 0
        calibration_offset_pitch = 0
    
    return jsonify({"status": "calibration_reset", "message": "Look at center of screen and press 'c' to calibrate"})


@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    if 'emergency_contact' in data:
        SETTINGS['emergency_contact'] = data['emergency_contact']
    if 'cursor_scope' in data:
        try:
            SETTINGS['cursor_scope'] = float(data['cursor_scope'])
        except:
            pass
    if 'blink_sensitivity' in data:
        try:
            SETTINGS['blink_sensitivity'] = float(data['blink_sensitivity'])
        except:
            pass
    if 'cursor_speed' in data:
        try:
            speed = float(data['cursor_speed'])
            SETTINGS['cursor_speed'] = max(0.3, min(1.5, speed))  # Clamp to valid range
        except:
            pass
    if 'overlay_enabled' in data:
        SETTINGS['overlay_enabled'] = bool(data['overlay_enabled'])
    if 'mouse_control_enabled' in data:
        SETTINGS['mouse_control_enabled'] = bool(data['mouse_control_enabled'])
    return jsonify({"status": "updated", "settings": SETTINGS})


@app.route('/perform_action', methods=['POST'])
def perform_action():
    global calibration_offset_yaw, calibration_offset_pitch
    
    data = request.json
    action = data.get('action')
    text = data.get('text')

    contact_num = SETTINGS["emergency_contact"]
    msg_body = "SOS! I need help. Sent via Sensceway."

    if action == 'google':
        webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(text)}")
    elif action == 'youtube':
        webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(text)}")
    elif action == 'type_external':
        threading.Thread(target=execute_type_external, args=(text,)).start()
        return jsonify({"status": "timer_started"})

    elif action == 'emergency_contact':
        threading.Thread(target=auto_send_whatsapp, args=(contact_num, msg_body)).start()

    elif action == 'dial_contact':
        webbrowser.open(f"tel:{contact_num}")
        return jsonify({"status": "dialer_opened"})

    elif action == 'emergency_call':
        threading.Thread(target=make_twilio_call).start()

    elif action == 'emergency_police':
        webbrowser.open("tel:100")
    elif action == 'emergency_ambulance':
        webbrowser.open("tel:112")
        
    elif action == 'toggle_overlay':
        SETTINGS['overlay_enabled'] = not SETTINGS.get('overlay_enabled', True)
        return jsonify({"status": "toggled", "overlay_enabled": SETTINGS['overlay_enabled']})
        
    elif action == 'toggle_mouse':
        SETTINGS['mouse_control_enabled'] = not SETTINGS.get('mouse_control_enabled', True)
        return jsonify({"status": "toggled", "mouse_control_enabled": SETTINGS['mouse_control_enabled']})

    return jsonify({"status": "ok"})


# --- FATIGUE MONITORING ROUTES ---
@app.route('/fatigue_status')
def fatigue_status():
    """Get current fatigue monitoring status for UI polling."""
    status = fatigue_monitor.check_fatigue()
    return jsonify(status)


@app.route('/fatigue_stats')
def fatigue_stats():
    """Get fatigue monitoring statistics for tuning."""
    stats = fatigue_monitor.get_stats()
    return jsonify(stats)


# --- VOICE COMMAND ROUTES ---
@app.route('/voice_command', methods=['POST'])
def voice_command():
    """
    Process a voice command from the frontend.
    
    Expects JSON: {"text": "search hello on google"}
    Returns action status or pending confirmation request.
    """
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'status': 'error', 'message': 'No command text provided'})
    
    result = voice_executor.process_command(text)
    return jsonify(result)


@app.route('/confirm_action', methods=['POST'])
def confirm_action():
    """Confirm the pending voice command action."""
    result = voice_executor.confirm_pending()
    return jsonify(result)


@app.route('/cancel_action', methods=['POST'])
def cancel_action():
    """Cancel the pending voice command action."""
    result = voice_executor.cancel_pending()
    return jsonify(result)


@app.route('/pending_action')
def pending_action():
    """Check if there's a pending action awaiting confirmation."""
    pending = voice_executor.get_pending()
    if pending:
        return jsonify({'has_pending': True, **pending})
    return jsonify({'has_pending': False})


# --- KEYBOARD LISTENER FOR CALIBRATION ---
def keyboard_listener():
    """Listen for keyboard input for calibration ('c' key)."""
    global calibration_offset_yaw, calibration_offset_pitch
    
    try:
        from pynput import keyboard
        from pynput.keyboard import Key
        
        def on_press(key):
            global calibration_offset_yaw, calibration_offset_pitch
            try:
                if hasattr(key, 'char') and key.char == 'c':
                    # Calibration will be handled in the next frame
                    # We need access to raw values, so we set a flag
                    print("[Calibration] Press detected - calibrating on next frame...")
            except AttributeError:
                pass
        
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        listener.join()
    except ImportError:
        print("WARNING: pynput not installed. Keyboard calibration disabled.")
        print("Install with: pip install pynput")


if __name__ == '__main__':
    # Start mouse mover thread
    threading.Thread(target=mouse_mover, daemon=True).start()
    
    # Start keyboard listener for calibration
    threading.Thread(target=keyboard_listener, daemon=True).start()
    
    # Run Flask in a background thread
    threading.Thread(target=lambda: app.run(port=5001, debug=False, use_reloader=False), daemon=True).start()
    
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:5001")
    
    # PyQt5 MUST run in the main thread on macOS
    if PYQT5_AVAILABLE:
        qt_app = QtWidgets.QApplication(sys.argv)
        overlay = CursorOverlay(radius=80)
        overlay.show()
        qt_app.exec_()
    else:
        # Keep the main thread alive if no Qt
        print("Running without cursor overlay. Press Ctrl+C to quit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
