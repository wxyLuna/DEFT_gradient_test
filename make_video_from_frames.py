import os
import subprocess
from glob import glob

def make_videos_from_frames(frame_dir="trajectory_plots_coupling", output_dir="trajectory_videos", fps=30):
    """
    Converts saved trajectory image frames into videos using ffmpeg.

    Args:
        frame_dir (str): Directory where frames are stored.
        output_dir (str): Directory to store the resulting videos.
        fps (int): Frames per second for the video.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all frame files
    frame_files = sorted(glob(os.path.join(frame_dir, "sample*_t*.png")))

    # Group frames by (sample, wire)
    from collections import defaultdict
    frame_dict = defaultdict(list)
    for path in frame_files:
        basename = os.path.basename(path)
        parts = basename.split("_")
        sample = parts[0][6:]  # after 'sample'
        wire = parts[1][4:]    # after 'wire'
        key = (sample, wire)
        frame_dict[key].append(path)


    # Create video for each group
    for (sample, wire), frames in frame_dict.items():
        print('hi')
        # Ensure sorting by timestep
        frames.sort()


        # Prepare a temporary directory with renamed sequential files
        tmp_dir = os.path.join(output_dir, f"temp_sample{sample}_wire{wire}")

        os.makedirs(tmp_dir, exist_ok=True)
        for i, src_path in enumerate(frames):
            dst_path = os.path.join(tmp_dir, f"frame_{i:03d}.png")
            os.system(f"cp {src_path} {dst_path}")

        # Generate video
        output_path = os.path.join(output_dir, f"{frame_dir }.mp4")
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%03d.png"),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        subprocess.run(cmd)

        # Optionally clean up temp frames
        for f in glob(os.path.join(tmp_dir, "*.png")):
            os.remove(f)
        os.rmdir(tmp_dir)

        print(f"Saved video to {output_path}")

if __name__ == "__main__":
    make_videos_from_frames()
