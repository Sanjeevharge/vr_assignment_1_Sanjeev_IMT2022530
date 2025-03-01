import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a folder to save output images
output_folder = r"C:\Users\sanje\Downloads\output_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Reload the image
image = cv2.imread('coins2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Use Hough Circle Transform to detect circular shapes (coins)
circles = cv2.HoughCircles(
    blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1.2, minDist=50, param1=50, param2=30, minRadius=20, maxRadius=100
)

# Draw detected circles on the image
output_image_hough = image.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw the outer circle
        cv2.circle(output_image_hough, (i[0], i[1]), i[2], (0, 255, 0), 3)
        # Draw the center of the circle
        cv2.circle(output_image_hough, (i[0], i[1]), 2, (0, 0, 255), 3)

# Count the total number of detected coins
coin_count_hough = len(circles[0, :]) if circles is not None else 0

# Save and display the final detected image
final_image_path = os.path.join(output_folder, "detected_coins.png")
cv2.imwrite(final_image_path, output_image_hough)

# Display the final detected coins using Hough Circle Transform
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(output_image_hough, cv2.COLOR_BGR2RGB))
plt.title(f"Final Detected Coins: {coin_count_hough} (Hough Circle Transform)")
plt.axis("off")
plt.show()

# Save and display segmented coin images
segmented_coins_hough = []
if circles is not None:
    for i, circle in enumerate(circles[0, :]):
        x, y, r = circle
        coin = image[y-r:y+r, x-r:x+r]  # Crop each detected coin
        segmented_coins_hough.append(coin)

        # Save each segmented coin image
        coin_image_path = os.path.join(output_folder, f"coin_{i+1}.png")
        cv2.imwrite(coin_image_path, coin)

# Display each segmented coin
for i, coin in enumerate(segmented_coins_hough):
    plt.figure(figsize=(2, 2))
    plt.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
    plt.title(f"Coin {i+1}")
    plt.axis("off")
    plt.show()

# Create and display a table for detected coins
df_hough = pd.DataFrame({"Hough Transform Detected Coins": [coin_count_hough]})
print(df_hough)
