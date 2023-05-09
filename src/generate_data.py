import csv
import os
import random
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from random_username.generate import generate_username
from rich.progress import Progress

from utils import filter_champions, filter_spells, load_champions, load_spells


def get_all_background_images(frames_root_folder):
    frames_root_path = Path(frames_root_folder)
    background_images_list = []

    for folder in frames_root_path.iterdir():
        if folder.is_dir():
            for img_file in folder.glob("*.jpg"):
                background_images_list.append(img_file)

    return background_images_list


def generate_empty_chat(background_images, output_path, max_width, max_height):
    for i in range(len(background_images)):
        random_background = random.choice(background_images)
        background = (
            Image.open(random_background)
            .resize((max_width, max_height))
            .convert("RGBA")
        )
        background.save(output_path / f"empty_chat_{i}.png")


def generate_no_spell_text(
    champions, spells, background_images, output_path, font, max_width, max_height
):
    for i in range(len(background_images)):
        random_teammate_champion = random.choice(champions)
        random_teammate_username = generate_username(1)[0]
        timestamp = f"[{random.randint(0, 59):02d}:{random.randint(0, 59):02d}]"
        text = f"{timestamp} {random_teammate_username} ({random_teammate_champion}): No spell pinged"

        random_background = random.choice(background_images)
        background = Image.open(random_background).resize((max_width, max_height))
        img = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 0))
        img.paste(background)

        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, (212, 215, 199), font=font)  # White color

        img.save(output_path / f"no_spell_text_{i}.png")


def generate_random_text(background_images, output_path, font, max_width, max_height):
    for i in range(len(background_images)):
        random_text = "".join([chr(random.randint(65, 90)) for _ in range(50)])

        random_background = random.choice(background_images)
        background = Image.open(random_background).resize((max_width, max_height))
        img = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 0))
        img.paste(background)

        draw = ImageDraw.Draw(img)
        draw.text((0, 0), random_text, (212, 215, 199), font=font)  # White color

        img.save(output_path / f"random_text_{i}.png")


def calculate_max_dimensions(champions, spells, font):
    max_width = 0
    max_height = 0
    for champion in champions:
        for spell in spells:
            text = f"[00:00] AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA ({champion}): {spell}"
            width, height = font.getsize(text)
            max_width = max(max_width, width)
            max_height = max(max_height, height)
    return max_width, max_height


def generate_synthetic_data(
    champions_csv,
    spells_csv,
    output_folder,
    background_images,
    num_champions,
    target_spells,
    num_images_per_combination=10,
    min_images_per_class=100,
):
    all_champions = load_champions(champions_csv)
    all_spells = load_spells(spells_csv)
    champions = filter_champions(all_champions, Path("data/generated").glob("**/*.png"))
    spells = filter_spells(all_spells, target_spells)

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use the Gill Sans font which is used in the in-game chat
    font = ImageFont.truetype("data/resources/Gill Sans.otf", 20)

    max_width, max_height = calculate_max_dimensions(champions, spells, font)

    actual_num_images_per_combination = max(
        num_images_per_combination, min_images_per_class
    )

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Generating synthetic data...",
            total=len(champions) * len(spells) * actual_num_images_per_combination,
        )

        for champion in champions:
            champion_output_path = output_path / champion
            champion_output_path.mkdir(parents=True, exist_ok=True)
            for spell in spells:
                for _ in range(actual_num_images_per_combination):
                    random_teammate_champion = random.choice(champions)
                    random_teammate_username = generate_username(1)[0]
                    timestamp = (
                        f"[{random.randint(0, 59):02d}:{random.randint(0, 59):02d}]"
                    )
                    text = f"{timestamp} {random_teammate_username} ({random_teammate_champion}): {champion} {spell}"

                    random_background = random.choice(background_images)
                    background = (
                        Image.open(random_background)
                        .resize((max_width, max_height))
                        .convert("RGBA")
                    )  # Convert background to RGBA
                    img = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 0))
                    img.paste(background)

                    draw = ImageDraw.Draw(img)

                    # Draw text with different colors for different components
                    draw.text(
                        (0, 0), timestamp, (212, 215, 199), font=font
                    )  # Timestamp color (white)
                    username_start = font.getlength(timestamp)
                    draw.text(
                        (username_start, 0),
                        f" {random_teammate_username} ({random_teammate_champion}): ",
                        (77, 167, 209),
                        font=font,
                    )  # Username and teammate champion color (blue)
                    champion_start = username_start + font.getlength(
                        f" {random_teammate_username} ({random_teammate_champion}): "
                    )
                    draw.text(
                        (champion_start, 0),
                        f"{champion} ",
                        (216, 55, 48),
                        font=font,
                    )  # Opponent champion color (red)
                    spell_start = champion_start + font.getlength(f"{champion} ")
                    draw.text(
                        (spell_start, 0), spell, (230, 171, 43), font=font
                    )  # Spell color (yellow)

                    img.save(champion_output_path / f"{text}.png")
                    progress.update(task, advance=1)
            # Generate non-matching synthetic data
    non_matching_output_path = output_path / "non_matching"
    non_matching_output_path.mkdir(parents=True, exist_ok=True)

    generate_empty_chat(
        background_images, non_matching_output_path, max_width, max_height
    )
    generate_no_spell_text(
        champions,
        spells,
        background_images,
        non_matching_output_path,
        font,
        max_width,
        max_height,
    )
    generate_random_text(
        background_images, non_matching_output_path, font, max_width, max_height
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", default="data/generated")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--num_champions", type=int, default=20)
    parser.add_argument(
        "--target_spells",
        nargs="+",
        default=("Heal", "Ghost", "Exhaust", "Flash", "Ignite"),
    )
    parser.add_argument("--min_images_per_class", type=int, default=200)

    args = parser.parse_args()

    generate_synthetic_data(
        "data/resources/champions.csv",
        "data/resources/spells.csv",
        args.output_dir,
        get_all_background_images("data/create_data/output"),
        args.num_champions,
        args.target_spells,
        num_images_per_combination=args.num_images,
        min_images_per_class=args.min_images_per_class,
    )
