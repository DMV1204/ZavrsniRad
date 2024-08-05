import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Nismo dodavali komentare jer su napisani u "NAOrobot_naivna_metoda.py"

new_image_path_v2 = "edges_camImage.png"
new_image_v2 = Image.open(new_image_path_v2).convert("L")

new_image_array_v2 = np.array(new_image_v2)

new_binary_array_v2 = np.where(new_image_array_v2 > 128, 1, 0)

new_white_pixels_v2 = np.argwhere(new_binary_array_v2 == 1)

#MEDIAN ZA CENTAR
new_center_x_v2 = np.median(new_white_pixels_v2[:, 1])
new_center_y_v2 = np.median(new_white_pixels_v2[:, 0])
new_center_v2 = (new_center_x_v2, new_center_y_v2)

new_radii_v2 = np.sqrt((new_white_pixels_v2[:, 1] - new_center_x_v2) ** 2 + (new_white_pixels_v2[:, 0] - new_center_y_v2) ** 2)

new_mean_radius_v2 = np.mean(new_radii_v2)

new_output_image_v2 = np.copy(new_binary_array_v2)

new_output_image_v2 = new_output_image_v2 * 255

fig, ax = plt.subplots()
ax.imshow(new_output_image_v2, cmap='gray')

new_circle_v2 = plt.Circle((new_center_x_v2, new_center_y_v2), new_mean_radius_v2, color='cyan', fill=False)
ax.add_patch(new_circle_v2)

plt.axis('off')

new_output_path_v2 = "fitted.png"
plt.savefig(new_output_path_v2, bbox_inches='tight', pad_inches=0)

distances = np.sqrt((new_white_pixels_v2[:, 1] - new_center_x_v2) ** 2 + (new_white_pixels_v2[:, 0] - new_center_y_v2) ** 2)

differences = distances - new_mean_radius_v2

mse = np.mean(differences ** 2)

threshold = 50.0

is_circle = mse < threshold

print(mse, is_circle)