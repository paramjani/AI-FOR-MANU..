from ultralytics import YOLO
import cv2
import os
import csv
import threading
from datetime import datetime
from playsound import playsound

model = YOLO("models/best.pt")

# 🔉 Non-blocking alert sound
def play_alert_sound():
    try:
        threading.Thread(target=playsound, args=("alert.wav",), daemon=True).start()
    except Exception as e:
        print(f"Error playing sound: {e}")

# ✅ Safe window close to prevent re-opening
def safe_close_all_windows():
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Flush GUI events

# ✅ 1. For Uploaded Image Detection
def process_uploaded_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]
    names = model.names
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    person_boxes = [boxes[i] for i, c in enumerate(classes) if names[c] == "person"]
    helmet_boxes = [boxes[i] for i, c in enumerate(classes) if names[c] == "helmet"]
    mask_boxes   = [boxes[i] for i, c in enumerate(classes) if names[c] == "mask"] if "mask" in names.values() else []

    annotated = results.plot()

    os.makedirs("violations", exist_ok=True)
    log_path = "violations/violation_log.csv"
    if not os.path.exists(log_path):
        with open(log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Violation", "Frame Saved"])

    sound_played = False

    for pbox in person_boxes:
        has_helmet = any(iou_overlap(pbox, hbox) for hbox in helmet_boxes)
        has_mask = any(iou_overlap(pbox, mbox) for mbox in mask_boxes)

        violations = []
        if not has_helmet:
            violations.append("No Helmet")
        if not has_mask:
            violations.append("No Mask")

        if violations:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            frame_path = f"violations/frame_{timestamp}.jpg"
            cv2.imwrite(frame_path, img)

            with open(log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                for v in violations:
                    writer.writerow([timestamp, v, os.path.basename(frame_path)])

            if not sound_played:
                play_alert_sound()
                sound_played = True

    # Show result image with proper closing
    cv2.imshow("📸 PPE Detection Result", annotated)
    cv2.waitKey(0)
    safe_close_all_windows()

    result_path = os.path.join("static/uploads", "result_" + os.path.basename(image_path))
    cv2.imwrite(result_path, annotated)
    return result_path

# ✅ 2. For Live Feed Detection
def process_live_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    os.makedirs("violations", exist_ok=True)
    csv_path = "violations/violation_log.csv"

    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Violation", "Frame Saved"])

    print("📡 Live Detection Started (Press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        results = model(frame)[0]
        names = model.names
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        person_boxes = [boxes[i] for i, c in enumerate(classes) if names[c] == "person"]
        helmet_boxes = [boxes[i] for i, c in enumerate(classes) if names[c] == "helmet"]
        mask_boxes   = [boxes[i] for i, c in enumerate(classes) if names[c] == "mask"] if "mask" in names.values() else []

        annotated = results.plot()

        sound_played = False

        for pbox in person_boxes:
            has_helmet = any(iou_overlap(pbox, hbox) for hbox in helmet_boxes)
            has_mask = any(iou_overlap(pbox, mbox) for mbox in mask_boxes)

            violations = []
            if not has_helmet:
                violations.append("No Helmet")
            if not has_mask:
                violations.append("No Mask")

            if violations:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                frame_path = f"violations/frame_{timestamp}.jpg"
                cv2.imwrite(frame_path, frame)

                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for v in violations:
                        writer.writerow([timestamp, v, os.path.basename(frame_path)])

                if not sound_played:
                    play_alert_sound()
                    sound_played = True

                cv2.rectangle(annotated, (int(pbox[0]), int(pbox[1])), (int(pbox[2]), int(pbox[3])), (0, 0, 255), 2)
                cv2.putText(annotated, "⚠️ " + ", ".join(violations), (int(pbox[0]), int(pbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("🛡️ PPE Detection - Live Feed", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    safe_close_all_windows()
    print("🛑 Live Detection Ended.")

# ✅ IOU Overlap Check
def iou_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    iou = interArea / float(boxAArea)
    return iou > 0.2
