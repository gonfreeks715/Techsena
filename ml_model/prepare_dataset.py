import os, shutil, random
from pathlib import Path
import cv2

SOURCES    = ["../datasets/pistols", "../datasets/baggage", "../datasets/prohibited"]
OUTPUT     = "../datasets/cargo_final"

def enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

pairs = []
for src in SOURCES:
    for split in ["train","valid","val","test"]:
        img_dir = Path(src)/"images"/split
        lbl_dir = Path(src)/"labels"/split
        if not img_dir.exists(): continue
        for img in img_dir.glob("*.[jp][pn]g"):
            lbl = lbl_dir/(img.stem+".txt")
            if lbl.exists(): pairs.append((img, lbl))

print(f"Found {len(pairs)} total images")
random.shuffle(pairs)
n = len(pairs)
splits = {"train": pairs[:int(n*0.7)], "val": pairs[int(n*0.7):int(n*0.9)], "test": pairs[int(n*0.9):]}

for name, sp in splits.items():
    (Path(OUTPUT)/"images"/name).mkdir(parents=True, exist_ok=True)
    (Path(OUTPUT)/"labels"/name).mkdir(parents=True, exist_ok=True)
    for i,(img,lbl) in enumerate(sp):
        out = f"{name}_{i:05d}"
        frame = cv2.imread(str(img))
        if frame is not None:
            cv2.imwrite(str(Path(OUTPUT)/"images"/name/f"{out}.jpg"), enhance(frame))
        shutil.copy(lbl, Path(OUTPUT)/"labels"/name/f"{out}.txt")
    print(f"  {name}: {len(sp)} images")

# Write data.yaml
yaml = f"""path: {os.path.abspath(OUTPUT)}
train: images/train
val: images/val
test: images/test
nc: 7
names: ['gun','knife','explosive','drug_package','liquid','electronic_device','organic_mass']
"""
(Path(OUTPUT)/"data.yaml").write_text(yaml)
print("✅ data.yaml created")
print(f"✅ Dataset ready at: {OUTPUT}")