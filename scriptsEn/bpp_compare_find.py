import os
import re
import numpy as np

# 解析本地文件内容
def parse_file(file_path):
    data = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            print(f"原始行: {line}")  # 调试信息

            # 使用正则匹配 bpp 数值
            match = re.search(r"(\S+)\s+bpp\s+([\d\.]+)", line)
            if match:
                img_name = os.path.basename(match.group(1).strip().strip(","))  # 去掉逗号
                try:
                    bpp_value = float(match.group(2).strip())  # 提取 bpp 数值
                    data[img_name] = bpp_value
                except ValueError:
                    print(f"⚠️ 警告: 解析 {line} 时 bpp 转换失败")

    print(f"✅ 解析结果: {data}")  # 调试解析数据
    return data


# 仅匹配相同的文件名
def find_matching_bpp(original, generated):
    print("\n🔍 调试: 原始数据文件名", list(original.keys()))
    print("🔍 调试: 生成数据文件名", list(generated.keys()))

    matching_results = {}

    for img in original.keys():
        if img in generated:  # 只有文件名相同的才进行比较
            orig_bpp = original[img]
            gen_bpp = generated[img]
            diff = abs(orig_bpp - gen_bpp)
            matching_results[img] = (orig_bpp, gen_bpp, diff)

    return matching_results


# 解析 A.txt 和 B.txt
DiffEIC_file_path = "K:/Rs_papers/6-V2_Efficient Diffusion Model/实验数据/Plot/DiffEIC/Kodak/kodak_1_2_8_bpp_0.03748/output.txt"
b_file_path = "K:/Rs_papers/6-V2_Efficient Diffusion Model/实验数据/Plot/TCM/Kodak/kodak_0.00025_bpp0.0289_real/output.txt"

original_data = parse_file(DiffEIC_file_path)
generated_data = parse_file(b_file_path)

# 计算相同文件名的 bpp 差值
matching_bpp_results = find_matching_bpp(original_data, generated_data)

# 如果匹配结果为空，打印调试信息
if not matching_bpp_results:
    print("\n❌ 没有找到相同的文件名，请检查文件名格式！")

# 按 bpp 差值从小到大排序
sorted_results = sorted(matching_bpp_results.items(), key=lambda x: x[1][2])

# 输出匹配结果
print("\n--- 相同文件名的 bpp 匹配 ---")
for img, (orig_bpp, gen_bpp, diff) in sorted_results[0:15]:
    print(f" DiffEIC： {img}, bpp {np.round(orig_bpp, 4)}  TCM： {img}, bpp {np.round(gen_bpp,4)}")
