import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread('Logo Mock-up on Paper Free PSD.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("❌ Error: image not found")
    exit()

# Sampling
downsampled = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
upsampled = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

# Quantization
levels = 16
quantized = np.floor(image / (256 / levels)) * (256 / levels)
quantized = np.uint8(quantized)


image2 = cv2.imread('Round Signage Logo Mockup.png', cv2.IMREAD_GRAYSCALE)
if image2 is None:
    image2 = image.copy()
image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))

add_img = cv2.add(image, image2)
sub_img = cv2.subtract(image, image2)
mul_img = cv2.multiply(image, 1.5)
div_img = cv2.divide(image, 1.5)

plt.figure(figsize=(14,10))


plt.subplot(2,4,1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2,4,2)
plt.imshow(downsampled, cmap='gray')
plt.title("Downsampled")
plt.axis('off')

plt.subplot(2,4,3)
plt.imshow(upsampled, cmap='gray')
plt.title("Upsampled")
plt.axis('off')

plt.subplot(2,4,4)
plt.imshow(quantized, cmap='gray')
plt.title("Quantized")
plt.axis('off')

# Row 2
plt.subplot(2,4,5)
plt.imshow(add_img, cmap='gray')
plt.title("Addition")
plt.axis('off')

plt.subplot(2,4,6)
plt.imshow(sub_img, cmap='gray')
plt.title("Subtraction")
plt.axis('off')

plt.subplot(2,4,7)
plt.imshow(mul_img, cmap='gray')
plt.title("Multiplication")
plt.axis('off')

plt.subplot(2,4,8)
plt.imshow(div_img, cmap='gray')
plt.title("Division")
plt.axis('off')

plt.tight_layout()
plt.show()
