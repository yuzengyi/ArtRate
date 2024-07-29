import pandas as pd
from scipy.stats import spearmanr

# 文件路径
files = ['Three Response-two.xlsx']
expert_file = 'zhuanjia-two.xlsx'

# 读取专家评分
df_expert = pd.read_excel(expert_file)

# 初始化一个空的DataFrame来保存相关系数结果
results = pd.DataFrame()

# 遍历每个响应文件
for file in files:
    df_response = pd.read_excel(file)
    # 存储相关系数
    correlations = []
    # 计算每个维度的Spearman相关系数
    for column in ['造型创造力', '色彩感染力', '线条丰富性', '思维']:
        corr, _ = spearmanr(df_response[column], df_expert[column])
        correlations.append(corr)
    # 创建临时DataFrame用于当前文件的结果
    temp_df = pd.DataFrame([[file.replace('.xlsx', '')] + correlations], columns=['Response Type', '造型创造力', '色彩感染力', '线条丰富性', '思维'])
    # 使用concat方法添加到结果DataFrame
    results = pd.concat([results, temp_df], ignore_index=True)

# 将结果保存到Excel文件
results.to_excel('Spearman Correlation Results_two.xlsx', index=False)
print("结果已保存到 'Spearman Correlation Results_two.xlsx'")
