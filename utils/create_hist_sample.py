# """
#  If you find this code useful, please cite our paper:

#  Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN:
#  Controlling Colors of GAN-Generated and Real Images via Color Histograms."
#  In CVPR, 2021.

#  @inproceedings{afifi2021histogan,
#   title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via
#   Color Histograms},
#   author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
#   booktitle={CVPR},
#   year={2021}
# }
# """
# from pathlib import Path
# import os
# from RGBuvHistBlock import RGBuvHistBlock
# import torch
# from PIL import Image
# from torchvision import transforms
# from torchvision.utils import save_image
# import numpy as np
# from os.path import splitext, join, basename, exists

# base_dir = Path('autodl-tmp/SeaDiff-main')
# image_folder = base_dir / 'UIEB' / 'train' / 'input'
# output_folder = base_dir / 'UIEB' / 'train' / 'histo'
# # image_folder = 'autodl-tmp/SeaDiff-main/UIEB/train/input'
# # output_folder = 'autodl-tmp/SeaDiff-main/UIEB/train/histo'
# # if exists(output_folder) is False:
# #   os.mkdir(output_folder)

# output_folder.mkdir(parents=True, exist_ok=True)  # 这一行替换原有的if判断和os.mkdir调用

# torch.cuda.set_device(0)
# histblock = RGBuvHistBlock(insz=336, h=336,
#                            resizing='sampling',
#                            method='inverse-quadratic',
#                            sigma=0.02,
#                            device=torch.cuda.current_device())
# transform = transforms.Compose([transforms.Resize((336, 336)),
#                                 transforms.ToTensor()])

# image_names = os.listdir(image_folder)
# for filename in image_names:
#   print(filename)
#   img_hist = Image.open(os.path.join(image_folder, filename))
#   img_hist = torch.unsqueeze(transform(img_hist), dim=0).to(
#     device=torch.cuda.current_device())
#   histogram = histblock(img_hist)
#   # histogram = histogram.cpu().numpy()
#   save_image(histogram * 255, os.path.join(output_folder, filename))
#   # np.save(join(output_dir, basename(splitext(filename)[0]) + '.npy'), histogram)


from pathlib import Path
import os
import argparse
from RGBuvHistBlock import RGBuvHistBlock
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from os.path import splitext, join, basename, exists

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Create histogram samples')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for histograms')
args = parser.parse_args()

# 使用命令行参数而不是硬编码路径
image_folder = Path(args.input_dir)
output_folder = Path(args.output_dir)

# 确保输出目录存在（递归创建）
output_folder.mkdir(parents=True, exist_ok=True)

print(f"输入目录: {image_folder}")
print(f"输出目录: {output_folder}")
print(f"输入目录是否存在: {image_folder.exists()}")

# 检查输入目录是否存在
if not image_folder.exists():
    print(f"错误: 输入目录不存在: {image_folder}")
    print(f"当前工作目录: {os.getcwd()}")
    exit(1)

# 检查输入目录是否为空
image_names = [f for f in image_folder.iterdir() if f.is_file()]
if not image_names:
    print(f"警告: 输入目录为空: {image_folder}")
    exit(0)

torch.cuda.set_device(0)
histblock = RGBuvHistBlock(insz=336, h=336,
                           resizing='sampling',
                           method='inverse-quadratic',
                           sigma=0.02,
                           device=torch.cuda.current_device())
transform = transforms.Compose([transforms.Resize((336, 336)),
                                transforms.ToTensor()])

for file_path in image_folder.iterdir():
    if file_path.is_file():
        print(f"处理: {file_path.name}")
        try:
            img_hist = Image.open(file_path)
            img_hist = torch.unsqueeze(transform(img_hist), dim=0).to(
                device=torch.cuda.current_device())
            histogram = histblock(img_hist)
            output_path = output_folder / file_path.name
            save_image(histogram * 255, output_path)
            print(f"已完成: {file_path.name}")
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")
