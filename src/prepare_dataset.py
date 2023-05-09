import csv
import random
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from utils import filter_champions, filter_spells, load_champions, load_spells


def show_sample_images(
    dataloader, output_path, name="loader", num_images=5, start_index=0, padding=10
):
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < start_index:
            continue

        images = batch["image"]
        labels = batch["label"]
        timestamps = batch["timestamp"]
        usernames = batch["username"]
        teammate_champions = batch["teammate_champion"]
        opponent_champions = batch["opponent_champion"]
        spells = batch["spell"]

        combined_image = None
        for i in range(min(num_images, len(images))):
            image = images[i].permute(1, 2, 0).numpy()

            image = Image.fromarray((image * 255).astype(np.uint8))

            cropped_image = image.crop((0, 0, 450, image.height))

            if combined_image is None:
                combined_image = Image.new(
                    "RGB",
                    (
                        cropped_image.width * 2,
                        num_images * (cropped_image.height + padding) - padding,
                    ),
                    color=(255, 255, 255),
                )

            y_offset = i * (cropped_image.height + padding)
            combined_image.paste(cropped_image, (0, y_offset))

            label_text = (
                f"Label: {labels[i]}, "
                f"Opponent Champion: {opponent_champions[i]}, "
                f"Spell: {spells[i]}"
            )

            draw = ImageDraw.Draw(combined_image)
            text_x_offset = cropped_image.width + 10
            draw.text((text_x_offset, y_offset), label_text, fill=(0, 0, 0))

        plt.imshow(combined_image)
        plt.axis("off")
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        plt.savefig(
            output_path / f"sample_images_{current_time}_{name}.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=600,
        )

        return


class LoLImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        champions_csv="data/resources/champions.csv",
        spells_csv="data/resources/spells.csv",
        transform=None,
        max_images_per_class=2000,
        num_champions=50,
        target_spells=("Heal", "Ghost", "Exhaust", "Flash", "Ignite"),
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        all_champions = load_champions(champions_csv)
        all_spells = load_spells(spells_csv)
        self.champions = filter_champions(all_champions, self.root_dir.glob("**/*.png"))
        self.spells = filter_spells(all_spells, target_spells)
        self.max_images_per_class = max_images_per_class
        self.image_paths, self.non_matching_count = self.filter_images_by_teams(
            self.root_dir.glob("**/*.png")
        )
        self.label_pattern = re.compile(
            r"\[(\d{2}:\d{2})\] (.*?) \((.*?)\): (.*?) (.*?)\.png"
        )
        self.label_mapping = self.create_label_mapping(self.image_paths)

    def filter_images_by_teams(self, image_paths):
        filtered_image_paths = []
        counter = {}
        non_matching_count = 0
        label_pattern = re.compile(
            r"\[(\d{2}:\d{2})\] (.*?) \((.*?)\): (.*?) (.*?)\.png"
        )

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

                if opponent_champion in self.champions and spell in self.spells:
                    label = f"{opponent_champion}-{spell}"
                    if label not in counter:
                        counter[label] = 0

                    if counter[label] < self.max_images_per_class:
                        filtered_image_paths.append(path)
                        counter[label] += 1
            else:
                non_matching_count += 1
                filtered_image_paths.append(path)

        return filtered_image_paths, non_matching_count

    def __len__(self):
        return len(self.image_paths)

    def create_label_mapping(self, image_paths):
        labels = set()
        for path in image_paths:
            match = self.label_pattern.match(path.name)
            if match:
                _, _, teammate_champion, opponent_champion, spell = match.groups()
                label = f"{opponent_champion}-{spell}"
                labels.add(label)
        return {label: idx for idx, label in enumerate(sorted(list(labels)))}

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to image: {e}")
                raise e

        match = self.label_pattern.match(image_path.name)

        if match:
            (
                timestamp,
                username,
                teammate_champion,
                opponent_champion,
                spell,
            ) = match.groups()
            label = f"{opponent_champion}-{spell}"
            label = torch.tensor(self.label_mapping.get(label, -1), dtype=torch.long)
        else:
            label = torch.tensor(-1, dtype=torch.long)
            timestamp, username, teammate_champion, opponent_champion, spell = (
                "unknown",
            ) * 5
        result = {
            "image": image,
            "label": label,
            "timestamp": timestamp,
            "username": username,
            "teammate_champion": teammate_champion,
            "opponent_champion": opponent_champion,
            "spell": spell,
        }

        return result


def get_dataset_stats(dataset, subset):
    champion_count = {}
    spell_count = {}
    champion_spell_count = {}
    non_matching_count = 0
    total_images = 0

    for idx in subset.indices:
        sample = dataset[idx]
        label = sample["label"]
        opponent_champion = sample["opponent_champion"]
        spell = sample["spell"]

        if label == -1:
            non_matching_count += 1
            total_images += 1
            continue

        champion_spell_combo = f"{opponent_champion}-{spell}"

        champion_count[opponent_champion] = champion_count.get(opponent_champion, 0) + 1
        spell_count[spell] = spell_count.get(spell, 0) + 1
        champion_spell_count[champion_spell_combo] = (
            champion_spell_count.get(champion_spell_combo, 0) + 1
        )
        total_images += 1

    stats = {
        "champion_count": champion_count,
        "spell_count": spell_count,
        "champion_spell_count": champion_spell_count,
        "non_matching_count": non_matching_count,
        "total_images": total_images,
        "num_classes": len(champion_spell_count),
    }

    return stats


def print_stats(stats, split_name):
    print(f"---- {split_name} dataset statistics ----")
    print(f"Number of champions: {len(stats['champion_count'])}")
    print(f"Number of spells: {len(stats['spell_count'])}")
    print(
        f"Number of unique champion-spell combinations (classes): {stats['num_classes']}"
    )
    print(f"Non-matching cases: {stats['non_matching_count']}")
    print(f"Total images: {stats['total_images']}")

    print("\nImages per class (first 5):")
    counter = 0
    for combo, count in stats["champion_spell_count"].items():
        if counter >= 5:
            break
        print(f"{combo}: {count}")
        counter += 1
    print()


def initialize_splits(dataset):
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    return train_set, val_set, test_set


if __name__ == "__main__":
    seed_everything(42, workers=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    filtered_dataset = LoLImageDataset(root_dir="data/generated", transform=transform)
    train_set, val_set, test_set = initialize_splits(filtered_dataset)

    train_stats = get_dataset_stats(filtered_dataset, train_set)
    val_stats = get_dataset_stats(filtered_dataset, val_set)
    test_stats = get_dataset_stats(filtered_dataset, test_set)

    print_stats(train_stats, "Train")
    print_stats(val_stats, "Validation")
    print_stats(test_stats, "Test")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, num_workers=0)

    output_path = Path("data/samples")
    show_sample_images(
        train_loader,
        output_path=output_path,
        name="train",
        num_images=5,
        start_index=0,
    )
    show_sample_images(
        val_loader, output_path=output_path, name="val", num_images=5, start_index=0
    )
    show_sample_images(
        test_loader, output_path=output_path, name="test", num_images=5, start_index=0
    )
