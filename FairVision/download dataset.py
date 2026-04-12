import os
import csv
from datasets import load_dataset

# ✅ Force ALL Hugging Face storage to D:
os.environ["HF_HOME"] = "D:/fairface_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/fairface_cache/datasets"
os.environ["HF_HUB_CACHE"] = "D:/fairface_cache/hub"

# (Optional) disable symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load dataset
dataset = load_dataset("HuggingFaceM4/FairFace", "0.25")

# Output folder
base_dir = "D:/IJSE CAME/programs/FairVision/fairface_data"

# Only available splits
splits = dataset.keys()

for split in splits:
    print(f"Processing {split}...")

    img_dir = os.path.join(base_dir, split, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_path = os.path.join(base_dir, f"{split}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "age", "gender", "race"])

        for i, item in enumerate(dataset[split]):
            try:
                image = item["image"]

                filename = f"{i}.jpg"
                filepath = os.path.join(img_dir, filename)

                # Save image directly to D:
                image.save(filepath)

                writer.writerow([
                    filename,
                    item.get("age", ""),
                    item.get("gender", ""),
                    item.get("race", "")
                ])

            except Exception as e:
                print(f"Error at {split} index {i}: {e}")

print("Done ✅")