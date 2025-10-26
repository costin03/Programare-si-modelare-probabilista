import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

# Problema 2

np.random.seed(42)

size = 5
original_image = np.random.choice([0, 1], size=(size, size), p=[0.3, 0.7])

print("\nImagine originala (5x5):")
print(original_image)

noise_ratio = 0.1
num_noisy_pixels = int(size * size * noise_ratio)
noisy_image = original_image.copy()

noisy_indices = np.random.choice(size * size, num_noisy_pixels, replace=False)
for idx in noisy_indices:
    i, j = idx // size, idx % size
    noisy_image[i, j] = 1 - noisy_image[i, j]

print(f"\nImagine cu zgomot (Aproximativ {int(noise_ratio * 100)}% pixeli modificati):")
print(noisy_image)
print(f"Numar pixeli modificati: {num_noisy_pixels}")

print("\nPartea a) - Definirea Markov Network")

model = MarkovNetwork()

nodes = []
for i in range(size):
    for j in range(size):
        node_name = f"x_{i}_{j}"
        nodes.append(node_name)
        model.add_node(node_name)

print(f"Numar noduri (pixeli): {len(nodes)}")

edges = []
for i in range(size):
    for j in range(size):
        current = f"x_{i}_{j}"

        if i > 0:
            neighbor = f"x_{i - 1}_{j}"
            edges.append((current, neighbor))
        if j > 0:
            neighbor = f"x_{i}_{j - 1}"
            edges.append((current, neighbor))

model.add_edges_from(edges)
print(f"Numar muchii: {len(edges)}")

lambda_param = 1.5


def create_observation_factor(node, observed_value, lambda_val):
    values = np.zeros(2)
    for state in [0, 1]:
        values[state] = np.exp(-lambda_val * (state - observed_value) ** 2)

    return DiscreteFactor(
        variables=[node],
        cardinality=[2],
        values=values
    )


def create_smoothness_factor(node1, node2):
    values = np.zeros((2, 2))
    for state1 in [0, 1]:
        for state2 in [0, 1]:
            values[state1, state2] = np.exp(-(state1 - state2) ** 2)

    return DiscreteFactor(
        variables=[node1, node2],
        cardinality=[2, 2],
        values=values.flatten()
    )


factors = []

print(f"\nAdaugare factori de observatie (lambda={lambda_param})...")
for i in range(size):
    for j in range(size):
        node = f"x_{i}_{j}"
        observed = noisy_image[i, j]
        factor = create_observation_factor(node, observed, lambda_param)
        factors.append(factor)

print(f"Factori de observatie: {size * size}")

print(f"Adaugare factori de smoothness")
for edge in edges:
    factor = create_smoothness_factor(edge[0], edge[1])
    factors.append(factor)

print(f"Factori de smoothness: {len(edges)}")

model.add_factors(*factors)

print(f"\nModel valid: {model.check_model()}")

print("Partea b) - Estimare MAP cu Belief Propagation")

bp = BeliefPropagation(model)

print("\n")

denoised_image = np.zeros((size, size), dtype=int)

for i in range(size):
    for j in range(size):
        node = f"x_{i}_{j}"
        marginal = bp.query(variables=[node], show_progress=False)
        denoised_image[i, j] = 1 if marginal.values[1] > marginal.values[0] else 0


print("\nImagine originala:")
print(original_image)

print("\nImagine cu zgomot:")
print(noisy_image)

print("\nImagine denoised (MAP estimate):")
print(denoised_image)

original_flat = original_image.flatten()
noisy_flat = noisy_image.flatten()
denoised_flat = denoised_image.flatten()

accuracy_noisy = np.mean(noisy_flat == original_flat)
accuracy_denoised = np.mean(denoised_flat == original_flat)

print(f"\nAccuracy imagine cu zgomot: {accuracy_noisy * 100:.2f}%")
print(f"Accuracy imagine denoised: {accuracy_denoised * 100:.2f}%")
print(f"Imbunatatire: {(accuracy_denoised - accuracy_noisy) * 100:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0].axis('off')
for i in range(size):
    for j in range(size):
        axes[0].text(j, i, str(original_image[i, j]),
                     ha='center', va='center', color='red', fontsize=12)

axes[1].imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
axes[1].set_title(f'Noisy Image (~{int(noise_ratio * 100)}% noise)',
                  fontsize=14, fontweight='bold')
axes[1].axis('off')
for i in range(size):
    for j in range(size):
        color = 'red' if noisy_image[i, j] != original_image[i, j] else 'blue'
        axes[1].text(j, i, str(noisy_image[i, j]),
                     ha='center', va='center', color=color, fontsize=12)

axes[2].imshow(denoised_image, cmap='gray', vmin=0, vmax=1)
axes[2].set_title(f'Denoised (MAP) - Accuracy: {accuracy_denoised * 100:.1f}%',
                  fontsize=14, fontweight='bold')
axes[2].axis('off')
for i in range(size):
    for j in range(size):
        color = 'green' if denoised_image[i, j] == original_image[i, j] else 'red'
        axes[2].text(j, i, str(denoised_image[i, j]),
                     ha='center', va='center', color=color, fontsize=12)

plt.tight_layout()
plt.savefig('image_denoising_results.png', dpi=300, bbox_inches='tight')
print("\nImagine salvata in 'image_denoising_results.png'")
