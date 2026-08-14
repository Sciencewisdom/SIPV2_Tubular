"""
Generate synthetic Massachusetts Roads-style dataset.
Uses random graph-based road networks + aerial-like textures.
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm

ROOT_DIR = "/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/data/raw/mass_roads"

# Dataset sizes
SPLIT_SIZES = {'train': 200, 'valid': 30, 'test': 30}
IMG_SIZE = 1500


def generate_random_graph(n_nodes=80, min_dist=40, img_size=1500):
    """Generate a random planar graph representing road intersections."""
    nodes = []
    for _ in range(n_nodes * 3):
        if len(nodes) >= n_nodes:
            break
        x = random.randint(50, img_size - 50)
        y = random.randint(50, img_size - 50)
        if all(np.hypot(x - nx, y - ny) > min_dist for nx, ny in nodes):
            nodes.append((x, y))

    edges = []
    nodes_arr = np.array(nodes)
    for i, (x, y) in enumerate(nodes):
        dists = np.hypot(nodes_arr[:, 0] - x, nodes_arr[:, 1] - y)
        nearest = np.argsort(dists)[1:4]
        for j in nearest:
            if dists[j] < 300:
                edge = tuple(sorted((i, j)))
                if edge not in edges:
                    edges.append(edge)
    return nodes, edges


def draw_roads(nodes, edges, img_size=1500, road_width=6):
    """Draw roads as binary mask."""
    mask = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(mask)
    for i, j in edges:
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        draw.line([(x1, y1), (x2, y2)], fill=255, width=road_width)
    # Round intersections
    for x, y in nodes:
        r = road_width // 2 + 1
        draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    return np.array(mask)


def generate_aerial_texture(img_size=1500):
    """Generate aerial-like background texture."""
    # Base color (greenish/brownish terrain)
    base = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    base[:, :, 0] = 180  # R
    base[:, :, 1] = 190  # G
    base[:, :, 2] = 160  # B

    # Add Perlin-like noise patches
    scale = 4
    for _ in range(15):
        cx = random.randint(0, img_size)
        cy = random.randint(0, img_size)
        radius = random.randint(100, 400)
        color = [
            random.randint(120, 220),
            random.randint(130, 230),
            random.randint(100, 200),
        ]
        y, x = np.ogrid[:img_size, :img_size]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = dist < radius
        for c in range(3):
            base[:, :, c] = np.where(mask, color[c], base[:, :, c])

    # Add noise
    noise = np.random.normal(0, 8, base.shape).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return base


def generate_sample(idx, img_size=1500):
    """Generate one synthetic image + mask pair."""
    random.seed(idx + 42)
    np.random.seed(idx + 42)

    nodes, edges = generate_random_graph(
        n_nodes=random.randint(60, 120),
        min_dist=random.randint(30, 60),
        img_size=img_size,
    )
    road_width = random.randint(4, 10)
    mask = draw_roads(nodes, edges, img_size=img_size, road_width=road_width)

    texture = generate_aerial_texture(img_size=img_size)

    # Blend roads into texture (lighter gray for roads)
    road_color = np.array([210, 210, 210], dtype=np.uint8)
    mask_3ch = np.stack([mask // 255] * 3, axis=-1)
    img = texture * (1 - mask_3ch) + road_color * mask_3ch
    img = img.astype(np.uint8)

    # Add some "buildings" (dark rectangles) to test confidence gate
    for _ in range(random.randint(5, 15)):
        bx = random.randint(100, img_size - 200)
        by = random.randint(100, img_size - 200)
        bw = random.randint(30, 100)
        bh = random.randint(30, 100)
        img[by:by+bh, bx:bx+bw] = [100, 100, 110]

    return img, mask


def main():
    for split, n_samples in SPLIT_SIZES.items():
        sat_dir = os.path.join(ROOT_DIR, split, 'sat')
        map_dir = os.path.join(ROOT_DIR, split, 'map')
        os.makedirs(sat_dir, exist_ok=True)
        os.makedirs(map_dir, exist_ok=True)

        print(f"Generating {split}: {n_samples} samples")
        for i in tqdm(range(n_samples), desc=split):
            img, mask = generate_sample(i + hash(split) % 10000)

            # Filename pattern matching real dataset
            fname = f"synth_{split}_{i:04d}.tiff"

            Image.fromarray(img).save(os.path.join(sat_dir, fname))
            Image.fromarray(mask).save(os.path.join(map_dir, fname))

    print("Synthetic dataset generation complete.")
    print(f"Total samples: {sum(SPLIT_SIZES.values())}")


if __name__ == '__main__':
    main()
