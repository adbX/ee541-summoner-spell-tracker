import subprocess
from pathlib import Path

import ffmpeg
from tqdm.auto import tqdm


def get_frames(video_path, output_folder, frame_interval=10):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get video duration
    video_info = ffmpeg.probe(video_path)
    duration = float(video_info["format"]["duration"])

    # Calculate crop dimensions
    width = int(video_info["streams"][0]["width"])
    height = int(video_info["streams"][0]["height"])
    crop_width = width // 4
    crop_height = height // 2

    # Extract frames every 10 seconds
    timestamps = [i for i in range(0, int(duration), frame_interval)]

    output_files = []

    with tqdm(
        total=len(timestamps), desc=f"Extracting frames from {video_path.name}"
    ) as progress:
        for i, timestamp in enumerate(timestamps):
            output_file = output_path / f"{video_path.stem}_frame_{i:04d}.jpg"

            if output_file.exists():
                # print(f"Skipping {output_file} because it already exists")
                output_files.append(output_file)
                progress.update(1)
                continue

            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",  # hide ffmpeg banner
                    "-loglevel",
                    "panic",  # hide ffmpeg output
                    "-hwaccel",
                    "nvdec",
                    "-ss",
                    str(timestamp),
                    "-i",
                    str(video_path),
                    "-vf",
                    f"crop={crop_width}:{crop_height}:0:{crop_height}",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",  # Output quality
                    str(output_file),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            output_files.append(output_file)
            progress.update(1)
    return output_files
