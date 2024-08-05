import numpy as np
import cv2
import matplotlib.pyplot as plt


image_path = 'edges_camImage.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Pohranjujemo koordinate bijelih piksela
coords = np.column_stack(np.where(image > 0))

# Konstruiramo matricu B
x = coords[:, 1]
y = coords[:, 0]
B = np.column_stack((x**2 + y**2, x, y, np.ones(x.shape)))

# Pozivamo ugrađenu SVD funkciju
U, S, Vt = np.linalg.svd(B, full_matrices=False)

print(Vt)

# Rješenje je desni singularni vektor koji odgovara najmanjoj singularnoj vrijednosti
u = Vt[-1]

# Iz vektora u izvlačimo koeficijente a, b1, b2 i c
a = u[0]
b1 = u[1]
b2 = u[2]
c = u[3]

# Računamo centar i radijus kružnice po izvedenim formulama
z1 = -b1 / (2 * a)
z2 = -b2 / (2 * a)
radius = np.sqrt((b1**2 + b2**2 - 4 * a * c) / (4 * a**2))

print(f"Center: ({z1}, {z2})")
print(f"Radius: {radius}")

# Crtamo kružnicu
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
circle = plt.Circle((z1, z2), radius, color='red', fill=False)
ax.add_patch(circle)

plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

output_path = 'alg_metoda_fitted.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)

print(f"Fitted circle image saved to: {output_path}")

# Računamo MSE:
distances = np.sqrt((coords[:, 1] - z1) ** 2 + (coords[:, 0] - z2) ** 2)
differences = distances - radius
mse = np.mean(differences ** 2)

threshold = 50.0

is_circle = mse < threshold

print(f"MSE: {mse}, Is Circle: {is_circle}")
