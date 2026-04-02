import os
import cv2
import numpy as np


# ---- 噪声函数 ----
def add_gaussian_noise(img, mean=0, sigma=0.05):
    noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = img / 255.0 + noise
    noisy_img = np.clip(noisy_img, 0, 1)
    return (noisy_img * 255).astype(np.uint8)


def add_salt_pepper_noise(img, amount=0.05):
    output = np.copy(img)
    num_salt = int(np.ceil(amount * img.size * 0.5))
    num_pepper = int(np.ceil(amount * img.size * 0.5))

    # Salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    output[coords[0], coords[1]] = 255

    # Pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    output[coords[0], coords[1]] = 0

    return output


# ---- 主函数 ----
def add_noise_to_folder(input_dir, output_dir, noise_type="gaussian", **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = ['.png', '.jpg', '.jpeg', '.bmp']

    for filename in os.listdir(input_dir):
        if not any(filename.lower().endswith(ext) for ext in supported_exts):
            continue

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue

        if noise_type == "gaussian":
            sigma = kwargs.get("sigma", 0.05)
            noisy_img = add_gaussian_noise(img, sigma=sigma)
            # suffix = f"_gaussian_{sigma}"
        elif noise_type == "s&p":
            amount = kwargs.get("amount", 0.05)
            noisy_img = add_salt_pepper_noise(img, amount=amount)
            # suffix = f"_sp_{amount}"
        else:
            raise ValueError("Unsupported noise type")

        base_name, ext = os.path.splitext(filename)
        out_path = os.path.join(output_dir, base_name +  ext)
        cv2.imwrite(out_path, noisy_img)
        print(f"Saved: {out_path}")


# ---- 使用示例 ----
input_folder = "kodak"  # 输入文件夹路径
output_folder = "kodak_Gau_0.03"  # 输出文件夹路径

# 添加高斯噪声 (sigma=0.05)
add_noise_to_folder(input_folder, output_folder, noise_type="gaussian", sigma=0.03)

# 添加椒盐噪声 (amount=5%)
# add_noise_to_folder(input_folder, output_folder, noise_type="s&p", amount=0.05)
