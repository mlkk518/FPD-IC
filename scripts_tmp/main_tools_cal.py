import cv2
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import math
import lpips
# from ms_ssim_torch import ms_ssim
from pytorch_msssim  import ms_ssim
import torch
from argparse import ArgumentParser, Namespace

# 载入模型，选择使用VGG作为后端
lpips_vgg = lpips.LPIPS(net='vgg').cuda()  # 使用GPU

# 图像预处理
def image_loader(image_name):
    try:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图像大小
            transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
        ])
        image = Image.open(image_name).convert("RGB")
        image = transform(image).unsqueeze(0)  # 增加batch维
        return image.cuda()  # 将图像送到GPU
    except Exception as e:
        print(f"Error processing image: {image_name}. Skipping. Error: {e}")
        return None


# 计算LPIPS
def compute_lpips(image1_path, image2_path):
    image1 = image_loader(image1_path)
    image2 = image_loader(image2_path)
    distance = lpips_vgg(image1, image2, normalize=True)
    return distance.item()

# 计算两个文件夹中同名图片的LPIPS
def calculate_lpips_for_same_names(folder1_path, folder2_path):
    lpips_values = []
    # 获取文件夹1中的图片名列表
    image_names = os.listdir(folder1_path)
    for image_name in image_names:
        # 构造对应的图片路径
        image1_path = os.path.join(folder1_path, image_name)
        image2_path = os.path.join(folder2_path, image_name)
        if os.path.exists(image2_path):
            # 如果文件夹2中存在同名图片，则计算它们之间的LPIPS
            lpips_value = compute_lpips(image1_path, image2_path)
            # print("image1_path", image1_path, "LPIPS", lpips_value)
            lpips_values.append(lpips_value)
        else:
            print(f"Image '{image_name}' not found in folder2")
    return lpips_values

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_Y_psnr(img1, img2):
    # Convert images to YUV color space
    img1_yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
    img2_yuv = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)

    # Extract Y channel
    y_channel1 = img1_yuv[:, :, 0]
    y_channel2 = img2_yuv[:, :, 0]

    # Calculate MSE for Y channel
    mse = np.mean((y_channel1 - y_channel2) ** 2)

    if mse == 0:
        return float('inf')

    # Calculate PSNR for Y channel
    return 20 * math.log10(255.0 / math.sqrt(mse))
def calculate_PSNR_for_same_names(folder1_path, folder2_path):
    image_names = os.listdir(folder1_path)
    psnr_total = 0.0
    psnr_Y_total = 0.0
    MS_SSIM_total = 0.0
    num_images = 0
    for image_name in image_names:
        # print("folder1_path", folder2_path)
        # if "0.0008" in folder2_path: ##  获取0.0008 时的 指标
        # 构造对应的图片路径
        image1_path = os.path.join(folder1_path, image_name)
        image2_path = os.path.join(folder2_path, image_name)

        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        if os.path.exists(image2_path):
            psnr = calculate_psnr(img1, img2)
            psnr_Y = calculate_Y_psnr(img1, img2)
            original_image = np.array(img1).astype(np.float64)
            reconstructed_image = np.array(img2).astype(np.float64)

            original_image = torch.unsqueeze(torch.tensor(original_image), dim=0)
            reconstructed_image = torch.unsqueeze(torch.tensor(reconstructed_image), dim=0)
            reconstructed_image = reconstructed_image.transpose(3, 1)
            original_image = original_image.transpose(3, 1)

            MS_SSIM = ms_ssim(original_image, reconstructed_image)

            # print("Img : ", image_name)
            # print("PSNR: ", psnr)
            # print("MS_SSIM: ", MS_SSIM)
            # print(",", psnr)
            psnr_Y_total += psnr_Y
            psnr_total += psnr
            MS_SSIM_total += MS_SSIM.item()
            num_images += 1
    # print("num_images", num_images)
    # assert 1 < 0
    if num_images == 0:
        return 0, 0, 0
    else:
        return psnr_total/num_images, MS_SSIM_total/num_images, psnr_Y_total/num_images


def img_name_modify_fun(Path_need_modification = '', Replace_name = '_LQ_rec.png', New_name='.png'):
    # 遍历文件夹 A 中的图片文件
    for filename in os.listdir(Path_need_modification):
        # 检查文件是否是图片文件（以 .png 结尾）
        # if filename.endswith(Replace_name):
            # 构建文件的完整路径
            file_path = os.path.join(Path_need_modification, filename)
            # 构建新的文件名（去掉 "_GT.png" 后缀）
            new_filename = filename.replace(Replace_name, New_name)
            # 构建新的文件的完整路径
            new_file_path = os.path.join(Path_need_modification, new_filename)
            # 修改文件名
            os.rename(file_path, new_file_path)



def parse_args() -> Namespace:
    parser = ArgumentParser()
    # model parameters
    parser.add_argument(
        "--task",
        type=str,
        default="sr",
        choices=["sr", "face", "denoise", "unaligned_face"],
        help="Task you want to do. Ignore this option if you are using self-trained model.",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v2.1",
        choices=["v1", "v2", "v2.1", "custom"],
        help="DiffBIR model version.",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="kodak",
        help="Path to testing image",
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=0.00015,
        help="TCM compression rate!",
    )


    return parser.parse_args()

if __name__ == "__main__":
    metric = ["PSNR", "MS-SSIM", "FID", "LPIPS"]

    args = parse_args()
    lamb= args.lamb
    test_data = args.testset

    if test_data in ["kodak", "Tecnick", "CLIC"]:
        GT_img_path = f'/media/xjtu-ei/Disk_8T/LJH/Datasets/Natura_datasets/Test_imgs/DiffEIC_TestImgs/{test_data}'
        dec_img_path = f'/media/xjtu-ei/Disk_8T/LJH/Compression_mlkk/img_recover/DiffBIR-main/results_tmp/{test_data}_{lamb}'
    else:
        print("Inavaliable datasets!!! ")
        assert 1 < 0

    fid = 0
    dec_path = dec_img_path
    print("\n \n ################ filename ###############", dec_path)
    # 计算PSNR
    avg_psnr, avg_ms_ssim, avg_PSNR_Y = calculate_PSNR_for_same_names(GT_img_path, dec_path)
    # if "0.0008" in filename:
    lpips_vgg = calculate_lpips_for_same_names(GT_img_path, dec_path)

    # fid = fid_score.calculate_fid_given_paths([GT_img_path, dec_path],  batch_size=25, device='cuda', dims=2048, num_workers=0)

    print(" Results: lambda {} on the {} dataset".format(lamb, test_data))
    print('psnr/ms_ssim/lpips: \n', np.round(avg_psnr, 2), np.round(avg_ms_ssim, 4), np.round(np.mean(lpips_vgg), 4)) #, np.round(fid,4)


    print("\n------------")
        # else:
        #     print('avg_psnr/avg_ms_ssim/avg_PSNR_Y', np.round(avg_psnr,2), np.round(avg_ms_ssim,2), np.round(avg_PSNR_Y,2))









