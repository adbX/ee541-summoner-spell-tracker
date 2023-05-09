import argparse
import csv
import glob
import os
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from from_video.extract_frames import get_frames
from from_video.ocr_tesseract import perform_ocr, process_text
from prepare_dataset import load_champions, load_spells
from tqdm import tqdm


def process_frames(frame_files, champions, spells):
    ocr_results = []
    with ThreadPoolExecutor() as ocr_executor:
        ocr_futures = {
            ocr_executor.submit(perform_ocr, frame_file, champions, spells): frame_file
            for frame_file in frame_files
        }

        for ocr_future in tqdm(
            as_completed(ocr_futures),
            total=len(ocr_futures),
            desc="Performing OCR",
            leave=False,
            file=sys.stderr,
        ):
            text = ocr_future.result()
            frame_file = ocr_futures[ocr_future]
            processed_text = process_text(text, champions, spells)
            ocr_results.append({"image": str(frame_file), "text": processed_text})
    return ocr_results


def process_data(should_process_video=True):
    champions = load_champions("data/resources/champions.csv")
    spells = load_spells("data/resources/spells.csv")

    if should_process_video:
        video_files = [
            Path(video) for video in glob.glob(os.path.join(video_dir, "*.mkv"))
        ]

        ocr_results = []
        with ThreadPoolExecutor() as video_executor:
            video_futures = {
                video_executor.submit(
                    get_frames,
                    video,
                    os.path.join(output_folder, video.stem),
                    frame_interval,
                ): video
                for video in video_files
            }

            for video_future in tqdm(
                as_completed(video_futures),
                total=len(video_futures),
                desc="Processing videos",
                file=sys.stderr,
            ):
                frame_files = video_future.result()
                ocr_results.extend(process_frames(frame_files, champions, spells))

    else:
        frame_subfolders = [
            Path(frame_dir)
            for frame_dir in glob.glob(os.path.join(frames_root_folder, "*"))
            if Path(frame_dir).is_dir()
        ]
        all_frame_files = []
        for frame_folder in frame_subfolders:
            all_frame_files.extend(list(frame_folder.glob("*.jpg")))

        ocr_results = process_frames(all_frame_files, champions, spells)

    with open(ocr_results_file, "w") as csvfile:
        fieldnames = ["image", "text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in ocr_results:
            writer.writerow(row)


def main(args):
    process_data(should_process_video=args.process_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and/or perform OCR.")
    parser.add_argument(
        "--process-video", action="store_true", help="Enable video processing"
    )

    args = parser.parse_args()

    video_dir = "data/create_data/screenrecord"
    output_folder = "data/create_data/output"
    ocr_results_file = "data/create_data/output/labels.csv"
    frames_root_folder = "data/create_data/output"
    frame_interval = 10

    main(args)
