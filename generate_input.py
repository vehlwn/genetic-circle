import matplotlib.pyplot as plt
import numpy as np


radius = 3


def gen_points(center, n):
    sigma = radius / 3.0
    points_x = np.random.normal(center[0], sigma, n)
    points_y = np.random.normal(center[1], sigma, n)
    points = np.stack([points_x, points_y])
    return points


points = np.concatenate(
    [
        gen_points(
            (
                2.0,
                3.0,
            ),
            50,
        ),
        gen_points((12.0, 13.0), 60),
        gen_points((20.0, 13.0), 70),
    ],
    axis=1,
)
print(points.shape)
print(points)
plt.scatter(points[0, :], points[1, :], marker=".", s=5)

plt.savefig("example.png", dpi=300)

with open("input.txt", "wt") as f:
    f.write(f"{radius}\n")
    f.write(f"{points.shape[1]}\n")
    for i in range(points.shape[1]):
        f.write(f"{points[0, i]} {points[1, i]}\n")
