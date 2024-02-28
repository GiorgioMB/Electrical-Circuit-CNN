# This is for creating csv files from each xml

import csv
import xml.etree.ElementTree as ET
import os
import json


def read_labels(classes_json: str):
    with open(classes_json, "r") as f:
        label_map = json.load(f)
    return label_map


def xml_to_csv(xml_file, csv_file, label_map):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    with open(csv_file, "w", newline="") as csv_file:

        csv_writer = csv.writer(csv_file)
        headers = [
            "Label",
            "object",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "rotation",
            "text",
        ]
        csv_writer.writerow(headers)

        for obj in root.findall(".//object"):
            object_name = obj.find("name").text
            label = label_map.get(object_name, "unknown")
            bndbox = obj.find("bndbox")

            xmin = bndbox.find("xmin").text
            ymin = bndbox.find("ymin").text
            xmax = bndbox.find("xmax").text
            ymax = bndbox.find("ymax").text

            rotation_element = bndbox.find("rotation")
            rotation = rotation_element.text if rotation_element is not None else ""
            text_element = obj.find("text")
            text = text_element.text if text_element is not None else ""

            row = [label, object_name, xmin, ymin, xmax, ymax, rotation, text]
            csv_writer.writerow(row)


def process_drafter_folders(label_map_file):
    label_map = read_labels(label_map_file)
    drafter_folders = [f"drafter_{i}" for i in range(26)]

    for drafter_folder in drafter_folders:
        annotations_folder = os.path.join(drafter_folder, "annotations")
        csv_output_folder = os.path.join(drafter_folder, "csvs")

        os.makedirs(csv_output_folder, exist_ok=True)

        for xml_file in os.listdir(annotations_folder):
            if xml_file.endswith(".xml"):
                xml_file_path = os.path.join(annotations_folder, xml_file)
                csv_file_path = os.path.join(
                    csv_output_folder, f"{os.path.splitext(xml_file)[0]}.csv"
                )

                xml_to_csv(xml_file_path, csv_file_path, label_map)


if __name__ == "__main__":
    process_drafter_folders("classes.json")
