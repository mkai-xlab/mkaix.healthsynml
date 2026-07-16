import cv2
import os

img_path = "data/Knee X-ray Images/MedicalExpert-I/4Severe/SevereG4 (1).png"
if not os.path.exists(img_path):
    print("Test image not found!")
    exit(1)

# Load image
img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"Loaded image: {img_path} with shape {img.shape}")

# Define the artifact save path
artifact_dir = r"C:\Users\vietn\.gemini\antigravity-ide\brain\22d32bc5-5b9a-47bc-9c21-d7a5a17e1cb1"
os.makedirs(artifact_dir, exist_ok=True)

# For a typical MedicalExpert image (e.g. 300x162 containing two knees):
# We split the image vertically in the middle:
# Left half: x from 0 to w//2
# Right half: x from w//2 to w

# Let's calculate a tight square crop for the joint space of each half.
# Usually the joint space is located around the vertical center (h//2) and the horizontal center of each half.
# Let's crop a square region of size h x h (or slightly smaller, e.g. 0.85 * h) centered on each half's center.

crop_size = int(h * 0.9)
half_w = w // 2

# Left knee center
left_cx = half_w // 2
left_cy = h // 2
left_x1 = max(0, left_cx - crop_size // 2)
left_x2 = min(half_w, left_x1 + crop_size)
left_y1 = max(0, left_cy - crop_size // 2)
left_y2 = min(h, left_y1 + crop_size)

# Right knee center
right_cx = half_w + (half_w // 2)
right_cy = h // 2
right_x1 = max(half_w, right_cx - crop_size // 2)
right_x2 = min(w, right_x1 + crop_size)
right_y1 = max(0, right_cy - crop_size // 2)
right_y2 = min(h, right_y1 + crop_size)

# Crop
left_crop = img[left_y1:left_y2, left_x1:left_x2]
right_crop = img[right_y1:right_y2, right_x1:right_x2]

# Draw boxes on original image for visualization
visual_img = img.copy()
# Draw left box in Green
cv2.rectangle(visual_img, (left_x1, left_y1), (left_x2, left_y2), (0, 255, 0), 2)
cv2.putText(visual_img, "Left Knee ROI", (left_x1, left_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Draw right box in Blue
cv2.rectangle(visual_img, (right_x1, right_y1), (right_x2, right_y2), (255, 0, 0), 2)
cv2.putText(visual_img, "Right Knee ROI", (right_x1, right_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# Save images to artifacts directory
cv2.imwrite(os.path.join(artifact_dir, "original_with_crop_boxes.png"), visual_img)
cv2.imwrite(os.path.join(artifact_dir, "left_knee_perfect_crop.png"), left_crop)
cv2.imwrite(os.path.join(artifact_dir, "right_knee_perfect_crop.png"), right_crop)

print("Demo images cropped and saved to artifacts successfully!")
