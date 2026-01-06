from pathlib import Path

dir1 = Path(r"C:/Users/RLW1KOR/OneDrive - Bosch Group/AppliedAI/Perception_stack/driveindia-det2seg_image/data/final_dataset/images/train")
dir2 = Path(r"C:/Users/RLW1KOR/OneDrive - Bosch Group/AppliedAI/Perception_stack/driveindia-det2seg_image/data/final_dataset/labels/train")

# Get just file names (no subdirs)
files1 = [p.name[:-4] for p in dir1.iterdir() if p.is_file()]
files2 = [p.name[:-4] for p in dir2.iterdir() if p.is_file()]

print(len(files1), len(files2))


print("Only in dir1:")
for name in files1:
    if name not in files2:
        print(" ", name)