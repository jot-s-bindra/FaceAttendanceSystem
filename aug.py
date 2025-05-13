import os
import cv2
import numpy as np
import imgaug.augmenters as iaa

# Path to face database
db_path = "face_database/"

# Augmentation pipeline
aug = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50% chance to flip
    iaa.Affine(rotate=(-20, 20)),  # Rotate between -20 to 20 degrees
    iaa.Multiply((0.8, 1.2)),  # Change brightness
    iaa.GaussianBlur(sigma=(0.0, 1.5)),  # Add slight blur
])

# Process all images in database
for person in os.listdir(db_path):
    person_folder = os.path.join(db_path, person)
    if os.path.isdir(person_folder):  # Ensure it's a folder
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if unreadable
            
            # Apply augmentation
            augmented_imgs = [aug.augment_image(img) for _ in range(3)]  # Generate 3 variations

            # Save augmented images
            for i, aug_img in enumerate(augmented_imgs):
                new_img_name = f"{img_name.split('.')[0]}_aug{i}.jpg"
                cv2.imwrite(os.path.join(person_folder, new_img_name), aug_img)

print("✅ Image augmentation complete!")
