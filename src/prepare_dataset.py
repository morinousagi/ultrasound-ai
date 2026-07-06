from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "TN5000"

IMAGE_DIR = RAW_DIR / "JPEGImages"
XML_DIR = RAW_DIR / "Annotations"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
CROP_DIR = OUTPUT_DIR / "cropped"

CSV_PATH = OUTPUT_DIR / "metadata.csv"

MARGIN = 0.15

RANDOM_STATE = 42


# ==========================================================
# Expand bounding box by MARGIN
# ==========================================================

def expand_bbox(xmin, ymin, xmax, ymax, width, height, margin=MARGIN):
    bw = xmax - xmin
    bh = ymax - ymin

    # Expand bounding box by margin and stay wiithin img boundaries
    xmin = max(0, int(xmin - bw * margin))
    ymin = max(0, int(ymin - bh * margin))

    xmax = min(width, int(xmax + bw * margin))
    ymax = min(height, int(ymax + bh * margin))

    return xmin, ymin, xmax, ymax


# ==========================================================
# Main: create cropped images and metadata CSV
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    xml_files = sorted(XML_DIR.glob("*.xml"))

    print(f"\nFound {len(xml_files)} annotation files.\n")

    for xml_path in tqdm(xml_files):

        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text

        image_path = IMAGE_DIR / filename

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Cannot read {image_path}")
            continue

        h, w = image.shape[:2]

        objects = root.findall("object")

        if len(objects) == 0:
            continue

        obj = objects[0]

        label = int(obj.find("name").text)

        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        xmin, ymin, xmax, ymax = expand_bbox(
            xmin,
            ymin,
            xmax,
            ymax,
            w,
            h,
        )

        crop = image[ymin:ymax, xmin:xmax]

        crop_filename = filename

        crop_path = CROP_DIR / crop_filename

        cv2.imwrite(str(crop_path), crop)

        rows.append(
            {
                "filename": filename,
                "crop_path": f"cropped/{crop_filename}",
                "class_name": "benign" if label == 0 else "malignant",
                "label": label,
                "width": w,
                "height": h,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    df = pd.DataFrame(rows)

    # -------------------------------------------------------
    # Split metadata into train 80%, validation 10%, test 10% sets
    # -------------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=train_df["label"],
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    df = pd.concat([train_df, val_df, test_df])

    df = df.sort_values("filename")

    df.to_csv(CSV_PATH, index=False)

    print("\n========================================")
    print("Dataset preparation complete")
    print("========================================")

    print(f"\nMetadata saved to:\n{CSV_PATH}")

    print("\nSplit counts")

    print(df["split"].value_counts())

    print("\nClass counts")

    print(df["label"].value_counts())

    print("\nClass distribution per split")

    print(pd.crosstab(df["split"], df["label"]))


if __name__ == "__main__":
    main()