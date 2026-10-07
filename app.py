import os
import sys
import cv2
import time
import threading
import smtplib
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# ======== OPTIONAL winsound (Windows only) ========
try:
    import winsound
except Exception:
    winsound = None

# ================= CONFIG ================= #
IP_CAMERA_URL = "http://192.0.0.4:8080/video"  # your IP cam; falls back to webcam if unavailable
MAX_PEOPLE = 4
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

EMAIL_ENABLED = True  # set False while testing if you want
SENDER_EMAIL = "alert.Warning100@gmail.com"
SENDER_PASSWORD = "ijvhnjeqpohjgdnc"  # Gmail app password
RECIPIENT_EMAIL = "alert.Warning100@gmail.com"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERT_SOUND_FILE = os.path.join(BASE_DIR, "alert.wav")  # optional sound file
SCREENSHOT_FILE = os.path.join(BASE_DIR, "alert_screenshot.jpg")

# ================= ALERT FUNCTIONS ================= #
def play_alert_sound():
    """Play alert sound asynchronously if winsound & file available."""
    if winsound and os.path.exists(ALERT_SOUND_FILE):
        def _play():
            try:
                winsound.PlaySound(ALERT_SOUND_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as e:
                print(f"[Sound error] {e}")
        threading.Thread(target=_play, daemon=True).start()
        print(">>> Alert sound triggered")
    else:
        # Silent on non-Windows or if file missing
        pass

def send_email_alert(subject, message, image_path=None):
    """Send email alert with optional image attachment."""
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECIPIENT_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header("Content-Disposition", "attachment", filename=os.path.basename(image_path))
                msg.attach(img)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"[Email error] {e}")

# ================= YOLO MODEL ================= #
print("Loading YOLOv8 model (yolov8n.pt)...")
model = YOLO("yolov8n.pt")
print("YOLOv8 model loaded successfully!")

# ================= VIDEO CAPTURE ================= #
def open_capture():
    cap_try = cv2.VideoCapture(IP_CAMERA_URL)
    if not cap_try.isOpened():
        print(f"Cannot open IP camera at {IP_CAMERA_URL}, falling back to local webcam (index 0)")
        cap_try.release()
        cap_try = cv2.VideoCapture(0)
    # Set a reasonable resolution
    cap_try.set(cv2.CAP_PROP_FRAME_WIDTH,  680)
    cap_try.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap_try

cap = open_capture()

# ================= APP STATE ================= #
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

people_count = 0
alert_sent = False
alerts_log = []          # keeps recent alerts (simple in-memory log)
people_history = []      # rolling history for chart (server-side mirror)
HISTORY_MAX = 300        # ~10 minutes if you poll every 2s

state_lock = threading.Lock()

def safe_append_history(value):
    with state_lock:
        people_history.append({"t": int(time.time()), "v": int(value)})
        if len(people_history) > HISTORY_MAX:
            people_history.pop(0)

# ================= VIDEO GENERATOR ================= #
def gen_frames():
    """MJPEG stream yielding annotated frames."""
    global people_count, alert_sent, alerts_log, cap
    while True:
        success, frame = cap.read()
        if not success:
            # try to reopen capture gracefully
            print("Frame read failed; attempting to reopen camera...")
            cap.release()
            time.sleep(1.0)
            cap = open_capture()
            continue

        # Run tracking on the frame
        try:
            results = model.track(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                tracker="bytetrack.yaml",
                persist=True,
                classes=[0]  # person class only
            )
        except Exception as e:
            print(f"[YOLO error] {e}")
            continue

        annotated = results[0].plot()
        boxes = results[0].boxes
        curr_count = len(boxes) if boxes is not None else 0

        with state_lock:
            people_count = curr_count

        # Alert logic
        if curr_count > MAX_PEOPLE:
            status_text = f"ALERT! People: {curr_count} (> {MAX_PEOPLE})"
            color = (0, 0, 255)
            if not alert_sent:
                print(f">>> People detected: {curr_count}, exceeded limit {MAX_PEOPLE}")
                play_alert_sound()
                # Save screenshot
                try:
                    cv2.imwrite(SCREENSHOT_FILE, annotated)
                except Exception as e:
                    print(f"[Screenshot error] {e}")
                if EMAIL_ENABLED:
                    send_email_alert(
                        subject="Crowd Alert - Too Many People",
                        message=f"Number of people exceeded limit!\nCurrent: {curr_count} (Limit: {MAX_PEOPLE})",
                        image_path=SCREENSHOT_FILE if os.path.exists(SCREENSHOT_FILE) else None
                    )
                with state_lock:
                    alerts_log.append(f"⚠️ {time.strftime('%H:%M:%S')} ALERT: {curr_count} people detected")
                    alerts_log = alerts_log[-50:]
                    alert_sent = True
        else:
            status_text = f"People in frame: {curr_count}"
            color = (0, 255, 0)
            with state_lock:
                alert_sent = False

        # Overlay status
        try:
            cv2.putText(annotated, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        except Exception:
            pass

        # Update history
        safe_append_history(curr_count)

        # Encode and yield
        ret, buffer = cv2.imencode(".jpg", annotated)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# ================= ROUTES ================= #
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    with state_lock:
        return jsonify({
            "people": people_count,
            "alerts": alerts_log[-10:],
            "limit": MAX_PEOPLE
        })

@app.route("/history")
def history():
    with state_lock:
        return jsonify(people_history[-HISTORY_MAX:])

# ================= MAIN ================= #
if __name__ == "__main__":
    # Hint for Linux/macOS users if winsound is missing
    if winsound is None:
        print("Note: winsound not available (non-Windows). Sound alerts will be skipped.")
    # Start server
    app.run(host="0.0.0.0", port=5000, debug=True)
