import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
import torchvision
from torchvision.transforms.functional import to_pil_image

warnings.filterwarnings("ignore")

print(torch.cuda.is_available())


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()
    # return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--result", type=str, help="Path to a result")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument(
        "--real", action="store_true", default=False
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args

def convert_units(flops, params):
    flops_in_giga = flops / 1e9
    params_in_mega = params / 1e6
    return flops_in_giga, params_in_mega


def calculate_bpp(file_path, image_width, image_height):
    """
    计算 bpp（bits per pixel）

    :param file_path: 文件路径
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :return: bpp 值
    """
    # 获取文件大小（字节）
    file_size_bytes = os.path.getsize(file_path)

    # 将文件大小转换为 bits
    file_size_bits = file_size_bytes * 8

    # 计算图像像素总数
    total_pixels = image_width * image_height

    # 计算 bpp
    bpp = file_size_bits / total_pixels
    return bpp

def main(argv):
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg", "tif"]:
            img_list.append(file)
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
    net = net.to(device)
    net.eval()

    if 0:
        from thop import profile
        input_data = torch.randn(1, 3, 256, 256).to('cuda')

        flops, params = profile(net, inputs=(input_data,))
        flops_giga, params_mega = convert_units(flops, params)

        print(f"FLOPs: {flops_giga:.2f} G, Parameters: {params_mega:.2f} M")
        assert 1 < 0

    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    coding_total_time = 0
    decoding_total_time = 0
    total_time = 0
    dictory = {}
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)



    if args.real:
        net.update()

        weight_name = args.checkpoint.split('.')[1].split('/')[0]  #  Kodak
        data_test_name = args.data.split('/')[6]
        print("data_test_name", data_test_name)

        # weight_name = args.checkpoint.split('/')[-1].split('.')[1]  ## LoveDA
        # data_test_name = args.data.split('/')[5]
        # print("data_test_name", data_test_name, "weight_name", weight_name)
        # save_path = "./results/rec_img_" + str(weight_name) + "/"


        save_path = args.result  #f"./results_test/{data_test_name}/" + str(weight_name) + "/"
        save_path_bin = "results_tmp_bin/"  #f"./results_test/{data_test_name}/" + str(weight_name) + "/"
        # save_path = "./results/Urban_codebook_LQ_" + str(weight_name) + "/"
        # save_path = "./results/Train_codebook_LQ_" + str(weight_name) + "/"
        # print(" save path", weight_name, " Crop size", crop_size)
        os.makedirs(save_path, exist_ok=True)

        os.makedirs(save_path_bin , exist_ok=True)
        # crop_size = 256
        # print("Start testing crop_size \n", crop_size)
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])
        with open(save_path + 'output.txt', 'w') as file:
            for img_name in img_list:
                print("img_name == ", img_name)
                img_path = os.path.join(path, img_name)
                s = time.time()

                img = transform(Image.open(img_path).convert('RGB')).to(device)
                x = img.unsqueeze(0)
                x_padded, padding = pad(x, p)
                count += 1
                with torch.no_grad():
                    if args.cuda:
                        torch.cuda.synchronize()
                    out_enc = net.compress(x_padded)
                    e = time.time()
                    coding_total_time += (e - s)

                    # print(" out_enc  = ", out_enc["strings"])

                    if 1:
                        # 将列表中的字节流合并为一个字节流
                        y_strings = b''.join(out_enc["strings"][0])
                        z_strings = b''.join(out_enc["strings"][1])

                        # 保存到文件
                        with open(save_path_bin + img_name + "y_strings.bin", "wb") as f:
                            f.write(y_strings)

                        with open(save_path_bin + img_name + "z_strings.bin", "wb") as f:
                            f.write(z_strings)

                        # 加载文件
                        with open(save_path_bin + img_name + "y_strings.bin", "rb") as f:
                            y_strings_loaded = f.read()

                        with open(save_path_bin + img_name + "z_strings.bin", "rb") as f:
                            z_strings_loaded = f.read()

                        # # 验证数据一致性
                        assert y_strings_loaded == y_strings, "y_strings 数据不一致"
                        assert z_strings_loaded == z_strings, "z_strings 数据不一致"
                        # print("数据验证通过，保存和加载的数据一致")

                        y_bpp = calculate_bpp(save_path_bin+ img_name +  "y_strings.bin",  x.size(2), x.size(3))
                        z_bpp = calculate_bpp(save_path_bin + img_name + "z_strings.bin", x.size(2), x.size(3))
                        total_bpp = y_bpp + z_bpp
                        # print(f"Total bpp: {total_bpp:.4f}")


                    s = time.time()
                    out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                    if args.cuda:
                        torch.cuda.synchronize()
                    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                    e = time.time()
                    decoding_total_time += (e - s)


                    # if "jpg" in img_name:
                    #     # print("out_dec shape", out_dec["x_hat"].shape)
                    #     pil_image = to_pil_image(torch.squeeze(out_dec["x_hat"], dim=0))
                    #     img_name = img_name.replace(".png", ".jpg")  # 假设之前是准备保存为PNG格式，这里改成JPG格式的后缀
                    #     quality = 100  # 设置较高的质量参数，取值范围是1-100，数值越高质量越好，信息损失越小
                    #     # pil_image.save(save_path + img_name, "JPEG", quality=quality)
                    # else:
                    torchvision.utils.save_image(out_dec["x_hat"], save_path + img_name, nrow=1)

                    num_pixels = x.size(0) * x.size(2) * x.size(3)
                    # print("num_pixels", num_pixels)
                    # print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
                    # print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.4f}dB')
                    # print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
                    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

                    # print("bpp ", bpp)
                    Bit_rate += bpp
                    PSNR += compute_psnr(x, out_dec["x_hat"])
                    MS_SSIM += compute_msssim(x, out_dec["x_hat"])

                    file.write(img_name + " bpp " + str(round(bpp, 4)) + " PSNR: " + str(
                        round(compute_psnr(x, out_dec["x_hat"]), 2)) + "\n")

            file.write("Avg: bpp " + str(round(Bit_rate / count, 4)) + " PSNR: " + str(
                round(PSNR / count, 2)) + "\n")

    else:
        # weight_name = args.checkpoint.split('/')[-1].split('.')[1]  ## LoveDA
        # data_test_name = args.data.split('/')[9]  #[5]
        # print("data_test_name", data_test_name, "weight_name", weight_name)

        weight_name = args.checkpoint.split('/')[-1].split('.')[1]  ## open_image
        data_test_name = args.data.split('/')[4]
        print("data_test_name", data_test_name, "weight_name", weight_name)

        save_path =  args.result # f"./results_test/{data_test_name}/" + str(weight_name) + "/"
        os.makedirs(save_path, exist_ok=True)
        # assert  1 < 0
        # folder_path = "/media/xjtu-ei/Disk_8T/LJH/LIC/LIC_TCM-main/results_test/open-images-30W_crop_256/"
        # 获取文件夹中的所有文件
        # all_files = os.listdir(folder_path)

        with open(save_path + 'output.txt', 'w') as file:
            for img_name in img_list:
                # print(" img_name ===== ", img_name)
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path).convert('RGB')
                x = transforms.ToTensor()(img).unsqueeze(0).to(device)
                x_padded, padding = pad(x, p)
                count += 1
                with torch.no_grad():
                    if args.cuda:
                        torch.cuda.synchronize()
                    s = time.time()
                    out_net = net.forward(x_padded)
                    if args.cuda:
                        torch.cuda.synchronize()
                    e = time.time()
                    total_time += (e - s)
                    out_net['x_hat'].clamp_(0, 1)
                    out_net["x_hat"] = crop(out_net["x_hat"], padding)

                    # bpp = compute_bpp(out_net)
                    # psnr_val = compute_psnr(x, out_net["x_hat"])
                    # ms_ssim_val = compute_msssim(x, out_net["x_hat"])

                    # imgPath = "./rec_img/bpp_" + str(
                    #     round(bpp, 4)) + "_" + "PSNR_" + str(round(psnr_val, 2)) + "msssim_" + str(
                    #     round(ms_ssim_val, 4)) + img_name

                    # if "jpg" in img_name:
                    #     # print("out_dec shape", out_dec["x_hat"].shape)
                    #     pil_image = to_pil_image(torch.squeeze(out_net["x_hat"], dim=0))
                    #     img_name = img_name.replace(".png", ".jpg")  # 假设之前是准备保存为PNG格式，这里改成JPG格式的后缀
                    #     quality = 100  # 设置较高的质量参数，取值范围是1-100，数值越高质量越好，信息损失越小
                    #     pil_image.save(save_path + img_name, "JPEG", quality=quality)
                    # else:
                    torchvision.utils.save_image(out_net["x_hat"], save_path + img_name, nrow=1)

                    # print(" ################  img_name #########  ", img_name)
                    # print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
                    # print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.4f}dB')
                    # print(f'Bit-rate: {compute_bpp(out_net):.4f}bpp')
                    PSNR += compute_psnr(x, out_net["x_hat"])
                    MS_SSIM += compute_msssim(x, out_net["x_hat"])
                    Bit_rate += compute_bpp(out_net)
                    #
                    # file.write(img_name + " bpp " + str(round(compute_bpp(out_net), 4)) + " PSNR: " + str(
                    #     round(compute_psnr(x, out_net["x_hat"]), 2)) + "\n")
            file.write("Avg: bpp " + str(round(Bit_rate / count, 4)) + " PSNR: " + str(
                round(PSNR / count, 2)) + "\n")

    PSNR = PSNR / count
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    coding_total_time = coding_total_time / count
    decoding_total_time = decoding_total_time/ count
    # total_time = total_time / count
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    print(f'average_Bit-rate: {Bit_rate:.4f} bpp')
    print(f'average_en_time: {coding_total_time:.4f} s')
    print(f'average_de_time: {decoding_total_time:.4f} s')


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
    