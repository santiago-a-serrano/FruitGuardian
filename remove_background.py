from rembg import remove
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def remove_bg(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    for file in tqdm(src_dir.rglob('*.png')):
        img = Image.open(file)
        output = remove(img)
        output_path = dst_dir / file.relative_to(src_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_path)

remove_bg('dataset/dataset', 'dataset/no_background')