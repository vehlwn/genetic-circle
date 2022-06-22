import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

points_x = np.array([])
points_y = np.array([])
radius = 0.0
n = 0
with open("input.txt", "rt") as f:
    radius = float(f.readline())
    n = int(f.readline())
    for _ in range(n):
        split = f.readline().split(maxsplit=2)
        x = float(split[0])
        y = float(split[1])
        points_x = np.concatenate((points_x, [x]))
        points_y = np.concatenate((points_y, [y]))

points = [(points_x[i], points_y[i]) for i in range(n)]
std_x = points_x.std()
std_y = points_y.std()
print(f"{std_x=}")
print(f"{std_y=}")
print(f"{radius=}")

population_size = 100
max_iterations = 100


def objective(center_x, center_y):
    ret = tf.zeros([population_size])
    for (x, y) in points:
        delta_x = center_x - x
        delta_y = center_y - y
        ret += tf.cast(
            tf.math.sqrt(delta_x ** 2 + delta_y ** 2) <= radius, tf.float32
        )
    print(f"{ret=}")
    return -ret


initial_position = (points_x[0], points_y[0])
optim_results = tfp.optimizer.differential_evolution_minimize(
    objective,
    initial_position=initial_position,
    population_size=population_size,
    max_iterations=max_iterations,
    population_stddev=max([std_x, std_y]),
)
print(f"{optim_results.converged=}")
print(f"{optim_results.position=}")
print(f"{optim_results.objective_value=}")
print(f"{optim_results.num_iterations=}")

figure, axes = plt.subplots()
cc = plt.Circle(
    (optim_results.position[0], optim_results.position[1]), radius, fill=False
)
axes.scatter(
    points_x,
    points_y,
    marker=".",
    s=5,
)
axes.add_artist(cc)
axes.set_aspect(1)
plt.savefig("optim_results.png", dpi=300)
plt.show()
