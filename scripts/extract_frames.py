import argparse
from src.loaders.driveindia_loader import DriveIndiaLoader

def main(args):
    loader = DriveIndiaLoader(
        root=args.data_root,
        fps=args.fps
    )
    loader.extract_frames(args.out_dir)
    print("[✓] Frame extraction complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed/frames")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    main(args)
