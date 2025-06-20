import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class FLIRRGB2ThermalDataset(Dataset):
    def __init__(self, rgb_dir, thermal_dir, img_size=256):
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.img_size = img_size

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.png'))])
        thermal_files = sorted([f for f in os.listdir(thermal_dir) if f.lower().endswith(('.jpg', '.png'))])

        # Debug: print file counts and a few filenames
        print(f"Found {len(rgb_files)} RGB files and {len(thermal_files)} thermal files.")
        print("Sample RGB files:", rgb_files[:5])
        print("Sample thermal files:", thermal_files[:5])

        # Match files by basename (without extension)
        rgb_basenames = {os.path.splitext(f)[0]: f for f in rgb_files}
        thermal_basenames = {os.path.splitext(f)[0]: f for f in thermal_files}
        common_basenames = sorted(set(rgb_basenames.keys()) & set(thermal_basenames.keys()))

        print(f"Found {len(common_basenames)} matching pairs.")

        if not common_basenames:
            raise RuntimeError("No matching RGB and thermal image pairs found. Check folder structure and filenames.")

        self.rgb_files = [rgb_basenames[b] for b in common_basenames]
        self.thermal_files = [thermal_basenames[b] for b in common_basenames]

        self.transform_rgb = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.transform_thermal = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        thermal_path = os.path.join(self.thermal_dir, self.thermal_files[idx])
        rgb = Image.open(rgb_path).convert('RGB')
        thermal = Image.open(thermal_path).convert('L')
        rgb = self.transform_rgb(rgb)
        thermal = self.transform_thermal(thermal)
        return rgb, thermal