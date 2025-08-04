import cv2
from deepface import DeepFace

def get_age_range(age):
    if age < 13:
        return "0-12"
    elif age < 20:
        return "13-19"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    else:
        return "60+"

def draw_meme_box(frame, x, y, w, h, age, gender, emotion):
    age_range = get_age_range(age)
    label = f" Age: {age_range}\n Gender: {gender}\n Emotion: {emotion}"

    # Draw face rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Draw a filled box above the rectangle
    box_height = 60
    cv2.rectangle(frame, (x, y - box_height), (x + w, y), (0, 0, 0), -1)

    # Add text in the box
    cv2.putText(frame, f"Age: {age_range}", (x + 5, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Gender: {gender}", (x + 5, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Emotion: {emotion}", (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot access webcam.")
        return

    print("✅ Running... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)

            if not isinstance(results, list):
                results = [results]

            for face in results:
                region = face.get("region", {})
                if not region:
                    continue

                x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 100), region.get('h', 100)
                age = face.get("age", 0)
                gender = face.get("gender", "Unknown")
                emotion = face.get("dominant_emotion", "Unknown")

                draw_meme_box(frame, x, y, w, h, age, gender, emotion)

        except Exception as e:
            print(f"⚠️ Error during analysis: {e}")

        cv2.imshow("SmartFaceAnalyzer - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
