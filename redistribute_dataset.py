
import sys
sys.path.append('D:\\PyCharm\\summer project\\split-folders-main')
import splitfolders
from pathlib import Path
import shutil

def merge_folders(source_dirs, merged_dir):
    dest_path = Path(merged_dir)
    dest_path.mkdir(parents=True, exist_ok=True) #创建目录

    for source_dir in source_dirs:
        source_path = Path(source_dir)
        for cls in source_path.iterdir():# iterdir()像迭代器iterator，
            if cls.is_dir():
                for img in cls.iterdir():
                    if img.is_file() and img.suffix in ['.jpeg']:#所有数据集中的的图片后缀都是jpeg
                        cls_dest_dir = dest_path / cls.name
                        cls_dest_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy(img, cls_dest_dir / img.name)


# 原始数据集路径
source_dirs = ['chest_xray/train', 'chest_xray/val', 'chest_xray/test']  #list

# 合并后的数据集路径
merged_dir = 'all_data'

# 合并文件夹
merge_folders(source_dirs, merged_dir)
# 目标文件夹路径
dest_dir = 'dataset'

# 将图片按7:2:1的比例分配到新的文件夹中
splitfolders.ratio(merged_dir, output=dest_dir, seed=1337, ratio=(.7, .2, .1))# seed用来保持每次运行代码的重复性

print("Images have been redistributed successfully!")
























