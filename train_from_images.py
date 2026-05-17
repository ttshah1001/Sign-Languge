#!/usr/bin/env python3
"""Train sign_model_26letters.pkl from public ASL alphabet images."""

import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("sign_model_26letters.pkl")
DATASET_ID = "Marxulia/asl_sign_languages_alphabets_v03"


def extract_landmarks(image_bgr, hands):
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if not results.multi_hand_landmarks:
        return None

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

    return np.array(landmarks, dtype=np.float32)


def build_dataset(samples_per_letter=150, split="train"):
    print(f"Loading {DATASET_ID} ({split})...")
    dataset = load_dataset(DATASET_ID, split=split)
    label_names = dataset.features["label"].names

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    X = []
    y = []
    skipped = 0
    per_letter_count = {name: 0 for name in label_names}

    for idx, row in enumerate(dataset):
        label_name = label_names[row["label"]]
        if per_letter_count[label_name] >= samples_per_letter:
            continue

        image = np.array(row["image"].convert("RGB"))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        landmarks = extract_landmarks(image_bgr, hands)

        if landmarks is None:
            skipped += 1
            continue

        X.append(landmarks)
        y.append(row["label"])
        per_letter_count[label_name] += 1

        if (idx + 1) % 500 == 0:
            collected = sum(per_letter_count.values())
            print(f"  processed {idx + 1} images, collected {collected}, skipped {skipped}")

        if all(count >= samples_per_letter for count in per_letter_count.values()):
            break

    hands.close()

    print("\nSamples per letter:")
    for letter in label_names:
        print(f"  {letter}: {per_letter_count[letter]}")

    print(f"\nTotal: {len(X)} samples ({skipped} images skipped — no hand detected)")
    return np.array(X), np.array(y), label_names


def train_and_save(X, y, label_names, output_path=MODEL_PATH):
    labels_dict = {i: name for i, name in enumerate(label_names)}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    print(f"Training on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.3f}")
    print(classification_report(y_test, y_pred, target_names=label_names))

    model_data = {"model": model, "labels_dict": labels_dict}
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ASL model from image dataset")
    parser.add_argument(
        "--samples-per-letter",
        type=int,
        default=150,
        help="Max training images per letter (default: 150)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH,
        help="Output pickle path",
    )
    args = parser.parse_args()

    X, y, label_names = build_dataset(samples_per_letter=args.samples_per_letter)
    if len(X) < 100:
        raise SystemExit("Not enough samples with detected hands. Try lowering detection threshold.")

    train_and_save(X, y, label_names, args.output)


if __name__ == "__main__":
    main()
