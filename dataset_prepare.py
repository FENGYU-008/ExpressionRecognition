import pandas as pd

# 加载FER2013数据集
data = pd.read_csv('data/fer2013/fer2013.csv')

# 分割数据集
train_data = data[data['Usage'] == 'Training']
test_data = data[data['Usage'] == 'PrivateTest']

# 将训练和测试数据集保存到新的CSV文件
train_data.to_csv('data/fer2013/train.csv', index=False)
test_data.to_csv('data/fer2013/test.csv', index=False)

print("数据集已成功分离和保存!")
