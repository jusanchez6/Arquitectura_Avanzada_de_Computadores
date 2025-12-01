import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import os

from PIL import Image as PilImage
from IPython.display import Image as DispImage, display



os.makedirs("frames", exist_ok=True)

points = np.loadtxt("very_big_input.csv", delimiter=",")
print(f"Puntos cargados: {points.shape[0]} filas")

centroid_files = sorted(glob.glob("centroids_iter_*.csv"))
label_files = sorted(glob.glob("labels_iter_*.csv"))

num_iters = len(centroid_files)
print(f"Iteraciones detectadas: {num_iters}")

frames = []

for it in range(num_iters):

    centroids = np.loadtxt(centroid_files[it], delimiter=",")
    labels = np.loadtxt(label_files[it], dtype=int)

    plt.figure(figsize=(6, 6))

    # Colorear por labels
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=12, cmap='tab10')

    # Centroides
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='black',
        s=120,
        marker='o',
        label="Centroides"
    )

    plt.title(f"K-Means GPU – Iteración {it}")
    plt.grid(True)
    plt.legend()

    # Guardar frame
    frame_path = f"frames/frame_{it:03d}.png"
    plt.savefig(frame_path)
    plt.close()

    frames.append(imageio.v2.imread(frame_path))

def save_gif(frames, filename="kmeans_gpu.gif", seconds_per_frame=1):
    pil_frames = [PilImage.fromarray(f) for f in frames]

    duration_ms = int(seconds_per_frame * 1000)  # Pillow usa ms

    pil_frames[0].save(
        filename,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0
    )


save_gif(frames, "kmeans_gpu.gif", seconds_per_frame=1)

display(DispImage(filename="kmeans_gpu.gif"))
