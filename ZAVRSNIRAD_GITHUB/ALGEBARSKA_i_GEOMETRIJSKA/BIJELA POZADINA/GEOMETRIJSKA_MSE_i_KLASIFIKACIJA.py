import numpy as np
import cv2
import matplotlib.pyplot as plt

#Konstruiramo matricu Jacobijana i vektor reziduala
def jacobian_matrix(x, y, z1, z2, r):
    m = len(x)
    J = np.zeros((m, 3))
    for i in range(m):
        dx = z1 - x[i]
        dy = z2 - y[i]
        dist = np.sqrt(dx**2 + dy**2)
        J[i, 0] = dx / dist
        J[i, 1] = dy / dist
        J[i, 2] = -1
    return J

def residuals(x, y, z1, z2, r):
    return np.sqrt((x - z1)**2 + (y - z2)**2) - r

# Učitavamo edge-detection sliku
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

# Rješenje je desni singularni vektor koji odgovara najmanjoj singularnoj vrijednosti

u = Vt[-1]

# Iz vektora u izvlačimo koeficijente a, b1, b2 i c
a = u[0]
b1 = u[1]
b2 = u[2]
c = u[3]

# Računamo centar i radijus kružnice po izvedenim formulama koji će nam biti početni parametri za Gauss-Newtonovu metodu
z1 = -b1 / (2 * a)
z2 = -b2 / (2 * a)
r = np.sqrt((b1**2 + b2**2 - 4 * a * c) / (4 * a**2))

# Gauss-Newtonova methoda
tolerance = 1e-6
max_iterations = 100
for iteration in range(max_iterations):
    J = jacobian_matrix(x, y, z1, z2, r)
    r_vec = residuals(x, y, z1, z2, r)
    delta = np.linalg.lstsq(J, -r_vec, rcond=None)[0]
    z1 += delta[0]
    z2 += delta[1]
    r += delta[2]
    if np.linalg.norm(delta) < tolerance:
        break

print(f"Refined center: ({z1}, {z2})")
print(f"Refined radius: {r}")

# Crtamo kružnicu
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
circle = plt.Circle((z1, z2), r, color='magenta', fill=False)
ax.add_patch(circle)

plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

output_path = 'geo_metoda_fitted.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)

print(f"Refined circle image saved to: {output_path}")

# Računamo MSE
distances = np.sqrt((coords[:, 1] - z1) ** 2 + (coords[:, 0] - z2) ** 2)
differences = distances - r
mse = np.mean(differences ** 2)

threshold = 50.0

is_circle = mse < threshold

print(f"MSE: {mse}, Is Circle: {is_circle}")
