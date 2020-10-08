import os
import shutil
import sys

"""
数据集切割，比例为 2:2:1
"""

data_dir = '/home/juan/PycharmProjects/Trajectory-Transformer/datasets/radar'

os.chdir(data_dir)

files_list = os.listdir(data_dir)

files = list(filter(
    lambda f: not os.path.isdir(os.path.join(data_dir, f)), files_list))

# 文件夹检测
if 'train' in files or 'val' in files or 'test' in files:
    print("folders detected, exiting...")
    sys.exit(1)

cnt = 0

for i in files:
    if (cnt + 1) % 1000 == 0 or (cnt + 1) == len(files):
        print(str(cnt) + '/' + str(len(files)))
    if cnt < 6400:
        shutil.move(
            os.path.join(data_dir, i),
            os.path.join(data_dir, 'train'))
    elif 6400 <= cnt < 12800:
        shutil.move(
            os.path.join(data_dir, i),
            os.path.join(data_dir, 'val')
        )
    else:
        shutil.move(
            os.path.join(data_dir, i),
            os.path.join(data_dir, 'test')
        )
    cnt += 1
