# Intentix  
### Adaptive Motor-Accessible Human–Computer Interaction System  

---

## Overview  

Intentix is an adaptive assistive interaction system designed to enable individuals with motor disabilities to control digital devices using eye movement, structured blink confirmation, and voice commands.

The system supports users whose cognitive intent remains intact while motor execution becomes unstable or degraded due to conditions such as:

- Amyotrophic Lateral Sclerosis (ALS)  
- Parkinson’s disease  
- Stroke  
- Spinal cord injuries  
- Carpal Tunnel Syndrome (CTS)  
- Repetitive Strain Injuries (RSI)  

Intentix transforms noisy motor signals into validated, reliable digital actions using standard consumer hardware.

---

## Problem Statement  

Modern digital interfaces assume precise and sustained motor control. As motor ability declines, tasks such as cursor positioning, clicking, and typing become unreliable or inaccessible.

Motor disability exists on a spectrum of signal degradation. Most assistive technologies either require expensive specialized hardware or fail under real-world instability.

Intentix addresses this by designing for degraded motor signals rather than ideal physical precision.

---

## Key Features  

- Real-time eye-gaze tracking using a standard webcam  
- Cursor stabilization with tremor filtering  
- Multi-stage blink confirmation (Focus → Lock → Execute)  
- Voice-based task automation  
- Fatigue-aware blink monitoring  
- OS-level automation compatible with existing applications  
- Hardware-independent, low-cost deployment  

---

## System Architecture  

Intentix follows a modular layered architecture:

### I. Input Layer  
- Webcam (video stream)  
- Microphone (audio input)  

### II. Vision Processing Layer  
- MediaPipe Face Mesh for facial landmark detection  
- Iris coordinate mapping for gaze estimation  
- Eye Aspect Ratio (EAR) for blink detection  
- Exponential Moving Average smoothing for cursor stabilization  
- Dead-zone filtering for tremor tolerance  

### III. Intent Interpretation Layer  
- Multi-stage confirmation state machine  
- Voice command parsing  
- Hybrid signal validation  

### IV. Execution Layer  
- Cursor control  
- Mouse clicks  
- Keyboard input simulation  
- Task automation (email, alarm, search)  

### V. Feedback Layer  
- Visual focus indicators  
- Lock confirmation  
- Execution acknowledgment  
- Fatigue notifications  

---

## Technology Stack  

### Backend  
- Python  

### Computer Vision  
- MediaPipe Face Mesh  
- OpenCV  

### Automation  
- PyAutoGUI  

### Voice Processing  
- SpeechRecognition  
- Custom intent parsing logic  

### Frontend  
- Flask  
- HTML  
- CSS  
- JavaScript  

### Hardware  
- Standard webcam  
- Microphone  

---

## Real-Time Processing Flow  

1. Capture video frame  
2. Detect facial landmarks  
3. Estimate gaze direction  
4. Apply stabilization  
5. Detect blink  
6. Validate intent  
7. Execute system action  
8. Provide feedback  
9. Repeat  

---

## Installation  

### Prerequisites  

- Python 3.8 or higher  
- Webcam  
- Microphone  
- Windows / macOS / Linux  

---

### Clone Repository  

```bash
git clone https://github.com/your-username/intentix.git
cd intentix
