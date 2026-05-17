import os
import pickle
import threading
import time
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SignLanguageDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self._hands_lock = threading.Lock()
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.labels_dict = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
            6: "G",
            7: "H",
            8: "I",
            9: "J",
            10: "K",
            11: "L",
            12: "M",
            13: "N",
            14: "O",
            15: "P",
            16: "Q",
            17: "R",
            18: "S",
            19: "T",
            20: "U",
            21: "V",
            22: "W",
            23: "X",
            24: "Y",
            25: "Z",
        }

        self.letter_to_num = {v: k for k, v in self.labels_dict.items()}

        self.model = None
        self.prediction_history = []
        self.history_size = 8

    def _prepare_frame(self, image):
        if image is None or image.size == 0:
            return None

        frame = image.copy()
        height, width = frame.shape[:2]

        if width > 960:
            scale = 960 / width
            frame = cv2.resize(
                frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
            )

        if frame.mean() < 40:
            frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)

        return frame

    def extract_hand_landmarks(self, image):
        frame = self._prepare_frame(image)
        if frame is None:
            return None

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._hands_lock:
            results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            landmarks = []
            x_coords = []
            y_coords = []

            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)

            wrist_x, wrist_y = x_coords[0], y_coords[0]

            for i in range(len(x_coords)):
                normalized_x = x_coords[i] - wrist_x
                normalized_y = y_coords[i] - wrist_y
                landmarks.extend([normalized_x, normalized_y])

            return np.array(landmarks)

        return None

    def predict_from_image(self, image_bgr):
        landmarks = self.extract_hand_landmarks(image_bgr)
        if landmarks is None:
            return None, False
        if self.model is None:
            return None, True

        prediction = self.model.predict([landmarks])[0]
        smooth_prediction = self.smooth_predictions(prediction)
        letter = self.labels_dict.get(smooth_prediction, "Unknown")
        return letter, True

    def collect_data(self):
        cap = cv2.VideoCapture(0)

        print("Sign Language Data Collection - All 26 Letters")
        print("Instructions:")
        print("- Press any letter key (A-Z) to collect data")
        print("- Collect 30+ samples per letter for best results")
        print("- Press 'q' to quit, 's' to start training")
        print("- Press 'r' to see statistics")

        data = []
        labels = []
        collection_count = {letter: 0 for letter in self.labels_dict.values()}

        current_letter_display = None
        last_collection_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks = self.extract_hand_landmarks(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                cv2.rectangle(
                    frame, (5, 5), (frame.shape[1] - 5, frame.shape[0] - 5), (0, 255, 0), 3
                )
            else:
                cv2.rectangle(
                    frame, (5, 5), (frame.shape[1] - 5, frame.shape[0] - 5), (0, 0, 255), 3
                )

            cv2.putText(
                frame,
                "Press letter keys (A-Z) to collect data",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                "Q: quit  S: train  R: stats",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Total samples: {len(data)}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            if current_letter_display and time.time() - last_collection_time < 2:
                cv2.putText(
                    frame,
                    f"Collected: {current_letter_display}",
                    (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    3,
                )

            y_offset = 110
            sorted_letters = sorted(collection_count.items(), key=lambda x: x[1], reverse=True)
            for i, (letter, count) in enumerate(sorted_letters[:10]):
                cv2.putText(
                    frame,
                    f"{letter}: {count}",
                    (10, y_offset + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

            cv2.imshow("Sign Language Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key >= ord("a") and key <= ord("z"):
                letter = chr(key).upper()
            elif key >= ord("A") and key <= ord("Z"):
                letter = chr(key).upper()
            else:
                letter = None

            if letter and landmarks is not None:
                label_num = self.letter_to_num[letter]
                data.append(landmarks)
                labels.append(label_num)
                collection_count[letter] += 1
                current_letter_display = letter
                last_collection_time = time.time()
                print(f"Collected {letter}: {collection_count[letter]} samples")

            elif key == ord("q") or key == ord("Q"):
                break
            elif key == ord("s") or key == ord("S"):
                if len(data) >= 100:
                    print("Starting training...")
                    self.train_model(np.array(data), np.array(labels))
                    break
                else:
                    print(f"Need at least 100 samples total. Current: {len(data)}")
            elif key == ord("r") or key == ord("R"):
                self.show_statistics(collection_count)

        cap.release()
        cv2.destroyAllWindows()

    def show_statistics(self, collection_count):
        print("\n=== Collection Statistics ===")
        total = sum(collection_count.values())
        print(f"Total samples: {total}")
        print("Per letter breakdown:")

        for letter in self.labels_dict.values():
            count = collection_count[letter]
            status = "✓" if count >= 30 else "!" if count >= 15 else "✗"
            print(f"{letter}: {count:2d} {status}")

        print("✓ = Excellent (30+), ! = Good (15+), ✗ = Need more")
        print("=============================\n")

    def train_model(self, X, y):
        if len(X) < 100:
            print("Need at least 100 samples total.")
            return

        print(f"Training with {len(X)} samples...")
        print(f"Classes: {len(np.unique(y))}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")

        model_data = {"model": self.model, "labels_dict": self.labels_dict}
        with open("sign_model_26letters.pkl", "wb") as f:
            pickle.dump(model_data, f)
        print("Model saved!")

    def load_model(self, model_path=None):
        model_path = model_path or os.environ.get(
            "MODEL_PATH", "sign_model_26letters.pkl"
        )
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            self.model = model_data["model"]
            self.labels_dict = model_data["labels_dict"]
            print(f"Model loaded! Supports: {list(self.labels_dict.values())}")
            return True
        except FileNotFoundError:
            print("Model file not found.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def smooth_predictions(self, prediction):
        self.prediction_history.append(prediction)

        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)

        if len(self.prediction_history) >= 5:
            return Counter(self.prediction_history).most_common(1)[0][0]
        return prediction

    def detect_realtime(self):
        if self.model is None:
            print("No model loaded!")
            return

        cap = cv2.VideoCapture(0)

        print("Real-time Detection - All 26 Letters")
        print("Controls: Q=quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks = self.extract_hand_landmarks(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                cv2.rectangle(
                    frame, (5, 5), (frame.shape[1] - 5, frame.shape[0] - 5), (0, 255, 0), 2
                )
            else:
                cv2.rectangle(
                    frame, (5, 5), (frame.shape[1] - 5, frame.shape[0] - 5), (0, 0, 255), 2
                )

            if landmarks is not None:
                prediction = self.model.predict([landmarks])[0]
                smooth_prediction = self.smooth_predictions(prediction)
                predicted_letter = self.labels_dict.get(smooth_prediction, "Unknown")

                cv2.putText(
                    frame,
                    f"Letter: {predicted_letter}",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 0),
                    4,
                )
            else:
                cv2.putText(
                    frame,
                    "Show your hand",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    3,
                )

            cv2.putText(
                frame,
                "Q: quit",
                (50, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            cv2.imshow("Sign Language Detection - 26 Letters", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break

        cap.release()
        cv2.destroyAllWindows()
