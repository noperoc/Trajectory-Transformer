import os
import shutil

# val 拿出3份，test 拿出 1 份
# 最终比例 8:1:1
base_number = 16000 // 10

datasets_folder = '/home/juan/PycharmProjects/Trajectory-Transformer/datasets/radar'

train_folder = os.path.join(datasets_folder, 'train')

validate_folder = os.path.join(datasets_folder, 'val')

test_folder = os.path.join(datasets_folder, 'test')

val_file_list = os.listdir(validate_folder)

test_file_list = os.listdir(test_folder)

# 切分
val_file_list = val_file_list[0: base_number * 3]
test_file_list = test_file_list[0: base_number]

# 移动
for file in val_file_list:
    src = os.path.join(validate_folder, file)
    trg = os.path.join(train_folder, file)
    shutil.move(src, trg)

for file in test_file_list:
    src = os.path.join(test_folder, file)
    trg = os.path.join(train_folder, file)
    shutil.move(src, trg)

# 统计最后结果
print(len(os.listdir(train_folder)))
print(len(os.listdir(validate_folder)))
print(len(os.listdir(test_folder)))
