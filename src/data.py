import random
import cv2 as cv
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def train_test(training_data, labels, x_test=None, y_test=None, random_state=0, test_size=0.2):
    random.seed(random_state)

    # Shuffle data and labels together
    combined_data = list(zip(training_data, labels))
    random.shuffle(combined_data)
    training_data[:], labels[:] = zip(*combined_data)

    if x_test is None:  # No need to check y_test separately
        # Split into train and test sets
        split_index = int(len(training_data) * (1 - test_size))
        x_train, y_train = training_data[:split_index], labels[:split_index]
        x_test, y_test = training_data[split_index:], labels[split_index:]
        return (x_train, y_train), (x_test, y_test)
    else:
        return (training_data, labels), (x_test, y_test)


def from_directory(folders: list, labels: list, input_size: tuple = (150, 150)):
    training_data = []
    tasks = []

    def process_image(img_path, label):
        img = cv.imread(img_path)
        if img is not None:
            # Uses input_size from outer scope
            img = cv.resize(img, input_size)
            img = img / 255.0  # Normalize
            return img, label
        return None  # Skip unreadable images

    with ThreadPoolExecutor() as executor:
        for folder, label in zip(folders, labels):
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                tasks.append(executor.submit(process_image, img_path, label))

        for task in tasks:
            result = task.result()
            if result:
                training_data.append(result)

    return training_data
