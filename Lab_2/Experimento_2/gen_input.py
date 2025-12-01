import random

N = 1000

with open("very_big_input.csv", "w") as f:
    for _ in range(N):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        f.write(f"{x},{y}\n")

print("Archivo data.csv generado con 1000 puntos.")