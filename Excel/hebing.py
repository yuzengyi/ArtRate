import pandas as pd
import os

# 需要生成的文件名对应
response_types = ["Zero Response", "One Response", "Three Response"]

# 初始化一个空的字典来存储各类型响应的数据
data_frames = {response: [] for response in response_types}

# 遍历文件夹中的所有文件
for i in range(1, 51):  # 假设文件名从 1_1.xlsx 到 1_50.xlsx
    file_name = f'1_{i}.xlsx'
    file_path = os.path.join(file_name)
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 使用 pandas 读取 Excel 文件
        df = pd.read_excel(file_path)
        # 添加文件来源列
        df['Source File'] = file_name
        # 根据响应类型过滤数据并存储
        for response in response_types:
            filtered_data = df[df['Unnamed: 0'] == response]
            if not filtered_data.empty:
                data_frames[response].append(filtered_data)

# 为每种响应类型创建一个单独的Excel文件
for response, data_list in data_frames.items():
    if data_list:
        # 合并同一响应类型的所有数据
        combined_data = pd.concat(data_list, ignore_index=True)
        output_file_path = os.path.join(f"{response}.xlsx")
        # 保存到新的Excel文件
        combined_data.to_excel(output_file_path, index=False)
        print(f'文件 {output_file_path} 已生成。')
    else:
        print(f'没有找到响应类型为 {response} 的数据。')
