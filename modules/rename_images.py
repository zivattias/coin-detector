import os
import uuid


def rename_images_to_uuid(directory):
    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            new_filename = str(uuid.uuid4()) + ".jpeg"
            # Rename the file
            os.rename(
                os.path.join(directory, filename), os.path.join(directory, new_filename)
            )
            print(f'Renamed "{filename}" to "{new_filename}"')


if __name__ == "__main__":
    directory = (
        "/Users/zivattias/Desktop/coin-detector-latest/datasets/coin_detector_test"
    )
    rename_images_to_uuid(directory)
