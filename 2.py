import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def stitch_images(image_path1, image_path2, output_folder=r"C:\Users\sanje\Downloads\stitched_results"):
    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read images
    img1 = cv2.imread('p.jpeg')
    img2 = cv2.imread('q.jpeg')
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Use SIFT to detect keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    # Display detected keypoints
    img1_keypoints = cv2.drawKeypoints(img1, keypoints1, None, color=(0,255,0))
    img2_keypoints = cv2.drawKeypoints(img2, keypoints2, None, color=(0,255,0))
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img1_keypoints, cv2.COLOR_BGR2RGB))
    plt.title("Keypoints in Image 1")
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, "keypoints1.jpg"))
    
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img2_keypoints, cv2.COLOR_BGR2RGB))
    plt.title("Keypoints in Image 2")
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, "keypoints2.jpg"))
    plt.show()
    
    # Use FLANN-based matcher for better accuracy
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Draw matches
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(12,6))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title("Matched Keypoints")
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, "matched_keypoints.jpg"))
    plt.show()
    
    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Get size of the output image
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    
    # Warp first image to align with the second
    panorama_width = width1 + width2
    panorama_height = max(height1, height2)
    result = cv2.warpPerspective(img1, H, (panorama_width, panorama_height))
    
    # Overlay second image onto the result
    result[0:height2, 0:width2] = img2
    
    # Save panorama
    panorama_path = os.path.join(output_folder, "stitched_panorama.jpg")
    cv2.imwrite(panorama_path, result)
    
    return result

# Provide image paths
image1_path = "image1.jpg"  # Replace with your first image file path
image2_path = "image2.jpg"  # Replace with your second image file path

# Stitch images
panorama = stitch_images(image1_path, image2_path)

# Display the final panorama
plt.figure(figsize=(15, 7))
plt.axis('off')
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.title("Stitched Panorama")
plt.show()
