import json


def parse_ground_truth(json_data):
    result = {}

    for item in json_data["data"]:
        result[item["id"]] = item["value"]

    return result


def read_and_parse_json(file_path: str):
    with open(file_path, "r") as file:
        json_data = json.load(file)

    parsed_data = parse_ground_truth(json_data)

    return parsed_data
