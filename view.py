import cv2
import numpy as np
from matplotlib import pyplot as plt

# # Load the mask
mask = cv2.imread("mask_1342.jpg", cv2.IMREAD_GRAYSCALE)

# Check unique pixel values
print("Unique values in mask:", np.unique(mask))

# Visualize it properly
plt.imshow(mask, cmap='gray')
plt.title('Car Mask')
plt.axis('off')
plt.show()



# image = cv2.imread("predicted_masks/mask_1360.jpg", cv2.IMREAD_GRAYSCALE)

# # Load original image
# # image = cv2.imread("train_masks/fff9b3a5373f_16.png")  # Adjust path
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Resize mask if needed
# mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

# # Threshold mask
# _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# # Create red overlay
# red_mask = np.zeros_like(image)
# red_mask[:, :, 0] = binary_mask  # Red channel

# # Overlay with transparency
# alpha = 0.5
# overlayed = cv2.addWeighted(image, 1.0, red_mask, alpha, 0)

# # Show overlayed image
# plt.imshow(overlayed)
# # plt.imshow(red_mask)
# plt.title('Overlayed Mask on Image')
# plt.axis('off')
# plt.show()

