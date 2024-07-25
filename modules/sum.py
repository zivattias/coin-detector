import os

import cv2
import numpy as np
import pandas as pd
import torch

from modules.json_parse import read_and_parse_json


class CoinDetector:
    model: torch.nn.Module = None

    # Coin values class map, see notes.json for further information
    coins_values_map = {0: 1, 1: 10, 2: 2, 3: 5}

    color_map = {
        0: (55, 103, 203),  # Blue for 1 Shekel
        1: (111, 72, 188),  # Purple for 10 Shekel
        2: (236, 55, 28),  # Orange for 2 Shekel
        3: (62, 195, 63),  # Green for 5 Shekel
    }

    def __init__(self, weights_path: str):
        print(
            "Initializing and loading model. This process may take a few seconds, please wait."
        )
        print("If this process takes more than a minute or two, please re-run sum.py!")

        model = torch.hub.load(
            "yolov5",
            "custom",
            source="local",
            path=weights_path,
        )

        # Toggle AgnosticNMS to retrieve highest-confidence detection per object, preventing 'double-detection'
        model.agnostic = True
        self.model = model

    def detect_one(self, image_path: str):
        results = self.model(image_path)
        detections = results.pandas().xyxy[0]

        self.save_image_with_detections(image_path, detections)

        return detections

    def detect_many(self, images: list[str]):
        return [self.detect_one(image) for image in images]

    def sum_one(self, df: pd.DataFrame) -> int:
        return df["class"].map(self.coins_values_map).sum()

    def sum_many(self, dfs: list[pd.DataFrame]) -> list[int]:
        return [self.sum_one(df) for df in dfs]

    def calculate_accuracy_one(self, predicted: int, ground_truth: int) -> float:
        normalized_difference = abs(ground_truth - predicted) / ground_truth
        accuracy_percentage = (1 - normalized_difference) * 100
        return accuracy_percentage

    def calculate_accuracy_many(
        self, predicted: list[int], ground_truth: list[int]
    ) -> list[float]:
        return [
            self.calculate_accuracy_one(p, g) for p, g in zip(predicted, ground_truth)
        ]

    def save_image_with_detections(self, image_path: str, detections: pd.DataFrame):
        image = cv2.imread(image_path)

        for _, row in detections.iterrows():
            x1, y1, x2, y2, cls = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
                int(row["class"]),
            )
            color = self.color_map.get(cls, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)

            # Label with matching color background and white text
            label = f'{self.coins_values_map.get(cls)}: {row["confidence"]:.2f}%'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
            cv2.rectangle(
                image, (x1, y1 - h), (x1 + 180, y1), color, -1
            )  # Background rectangle
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_TRIPLEX,
                1.1,
                (255, 255, 255),
                2,
            )  # White text

        _, filename = os.path.split(image_path)
        new_image_path = f"datasets/coin_detector_test/predictions/{filename}"
        cv2.imwrite(new_image_path, image)
        print(f"Saved image with predictions to: {new_image_path}")


coin_detector = CoinDetector(weights_path="yolov5/runs/train/stage_2/weights/best.pt")


# Images detection: exercises 4, 5, and 6
images = [
    f"datasets/coin_detector_test/{image}"
    for image in os.listdir("datasets/coin_detector_test")
    if image.endswith(".jpeg")
]

images_names = [image.split("/")[-1].split(".")[0] for image in images]

ground_truh_dict = read_and_parse_json("datasets/coin_detector_test/ground_truth.json")
ground_truth = [ground_truh_dict[image] for image in images_names]


detections = coin_detector.detect_many(images)
sums = coin_detector.sum_many(detections)
accuracies = coin_detector.calculate_accuracy_many(sums, ground_truth)

for image, sum, accuracy in zip(images_names, sums, accuracies):
    print(f"Image: {image}")
    print(f"Detected ILS: {sum}")
    print(f"Ground-Truth ILS: {ground_truh_dict[image]}")
    print(f"Accuracy: {accuracy:.2f}%", end="\n\n")

avg_accuracy = np.mean(accuracies)

print(f"Average accuracy on test set: {avg_accuracy:.2f}%")
