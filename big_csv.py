# This is for creating a single csv file with all the information from all xmls


import os
import csv
import xml.etree.ElementTree as ET
import json


def read_labels(classes_json: str):
    with open(classes_json, "r") as f:
        label_map = json.load(f)
    return label_map


def xml_to_csv(xml_file, csv_writer, label_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text

    for obj in root.findall(".//object"):
        object_name = obj.find("name").text
        label = label_mapping.get(object_name, "unknown")
        bndbox = obj.find("bndbox")
        xmin = bndbox.find("xmin").text
        ymin = bndbox.find("ymin").text
        xmax = bndbox.find("xmax").text
        ymax = bndbox.find("ymax").text
        rotation_element = bndbox.find("rotation")
        rotation = rotation_element.text if rotation_element is not None else ""
        text_element = obj.find("text")
        text = text_element.text if text_element is not None else ""

        row = [filename, label, object_name, xmin, ymin, xmax, ymax, rotation, text]
        csv_writer.writerow(row)


def process_drafter_folders(label_mapping_file):
    label_mapping = read_labels(label_mapping_file)

    drafter_folders = [f"drafter_{i}" for i in range(26)]
    output_folder = "main_csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_csv_path = os.path.join(output_folder, "combined_output.csv")

    with open(output_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)

        headers = [
            "filename",
            "label",
            "object",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "rotation",
            "text",
        ]
        csv_writer.writerow(headers)

        for drafter_folder in drafter_folders:
            annotations_folder = os.path.join(drafter_folder, "annotations")

            for xml_file in os.listdir(annotations_folder):
                if xml_file.endswith(".xml"):
                    xml_file_path = os.path.join(annotations_folder, xml_file)
                    xml_to_csv(xml_file_path, csv_writer, label_mapping)


if __name__ == "__main__":
    process_drafter_folders("classes.json")
