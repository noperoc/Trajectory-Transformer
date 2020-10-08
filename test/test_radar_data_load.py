import sys

sys.path.append('/home/juan/PycharmProjects/Trajectory-Transformer')

import baselineUtils

if __name__ == "__main__":
    train, _ = baselineUtils.create_old_3dim_dataset()
    print(len(train))
