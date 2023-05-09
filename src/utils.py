import csv
import random
from pathlib import Path
import re


def load_champions(champions_csv):
    champions = []
    with open(champions_csv, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            champions.append(row[0])
    return champions


def load_spells(spells_csv):
    spells = []
    with open(spells_csv, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            spells.append(row[0])
    return spells


def filter_champions(champions, image_paths):
    dataset_champion_names = set()
    label_pattern = re.compile(r"\[(\d{2}:\d{2})\] (.*?) \((.*?)\): (.*?) (.*?)\.png")

    for path in image_paths:
        match = label_pattern.match(path.name)

        if match:
            (
                timestamp,
                username,
                teammate_champion,
                opponent_champion,
                spell,
            ) = match.groups()

            dataset_champion_names.add(opponent_champion)

    dataset_champions = [
        champion for champion in champions if champion in dataset_champion_names
    ]

    return dataset_champions


def filter_spells(spells, target_spells):
    return [spell for spell in spells if spell in target_spells]
