import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3

class RealTimeClassifier:
    def __init__(self, model_path='sign_language_model.keras'):
        self.model = tf.keras.models.load_model(model_path)
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7)
        self.engine = pyttsx3.init()
        self.labels = {i: chr(65 + i) for i in range(25)}  # Labels A-Y, Z is not present

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        image = image.reshape(1, 28, 28, 1) / 255.0
        return image

    def predict(self, image):
        processed = self.preprocess(image)
        predictions = self.model.predict(processed)
        class_idx = np.argmax(predictions)
        confidence = predictions[0][class_idx]
        return self.labels.get(class_idx, "?"), confidence

    def text_to_speech(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def classify_from_camera(self):
        cap = cv2.VideoCapture(0)
        print("Starting real-time classification... Press 'q' to quit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.mp_hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                    x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                    y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                    y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                    x_min, x_max, y_min, y_max = map(int, [x_min-20, x_max+20, y_min-20, y_max+20])

                    hand_img = frame[max(y_min,0):min(y_max,h), max(x_min,0):min(x_max,w)]
                    
                    if hand_img.size == 0:
                        continue

                    label, confidence = self.predict(hand_img)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                    cv2.putText(frame, f'{label} ({confidence:.2f})', (x_min, y_min-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

                    # Trigger text-to-speech only if confidence is high
                    if confidence > 0.8:
                        self.text_to_speech('Gesture represents: '+label)

            cv2.imshow('Real-Time Sign Language Classification', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    classifier = RealTimeClassifier()
    classifier.classify_from_camera()
