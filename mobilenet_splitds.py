import os
import shutil
import random

source_dir = "29_classes"  # Your folder with 29 subfolders of dog breeds
dest_dir = "dataset"
train_ratio = 0.8  # 80% train, 20% valid

# Create output directories
train_dir = os.path.join(dest_dir, "train")
valid_dir = os.path.join(dest_dir, "valid")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Loop through each breed folder
for breed_name in os.listdir(source_dir):
    breed_path = os.path.join(source_dir, breed_name)
    if not os.path.isdir(breed_path):
        continue

    # Get all images in the breed folder
    images = [img for img in os.listdir(breed_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    # Compute split
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    valid_images = images[split_index:]

    # Create subfolders for each breed
    os.makedirs(os.path.join(train_dir, breed_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, breed_name), exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(breed_path, img), os.path.join(train_dir, breed_name, img))
    for img in valid_images:
        shutil.copy(os.path.join(breed_path, img), os.path.join(valid_dir, breed_name, img))

    print(f"✅ {breed_name}: {len(train_images)} train, {len(valid_images)} valid")

print("🎉 Dataset successfully split into 'dataset/train' and 'dataset/valid'")
