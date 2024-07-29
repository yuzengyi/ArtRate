import pandas as pd

# 加载数据
df1 = pd.read_excel('chatgpt4o.xlsx')
df2 = pd.read_excel('qwen_vl.xlsx')
df3 = pd.read_excel('zhuanjia.xlsx')

# 假设df3是参照组，我们将df1和df2与df3进行相关性计算
# 可以假设ID列是Source File，首先设置索引
df1.set_index('Source File', inplace=True)
df2.set_index('Source File', inplace=True)
df3.set_index('Source File', inplace=True)

# 计算皮尔逊相关性系数
pearson_corr_df1 = df1.corrwith(df3, method='pearson')
pearson_corr_df2 = df2.corrwith(df3, method='pearson')

# 计算斯皮尔曼相关性系数
spearman_corr_df1 = df1.corrwith(df3, method='spearman')
spearman_corr_df2 = df2.corrwith(df3, method='spearman')

# 创建一个新的DataFrame来保存结果
results = pd.DataFrame({
    'Pearson DF1': pearson_corr_df1,
    'Pearson DF2': pearson_corr_df2,
    'Spearman DF1': spearman_corr_df1,
    'Spearman DF2': spearman_corr_df2
})

# 保存到Excel
results.to_excel('correlation_results.xlsx')

print("相关性分析完成，结果已保存到 'correlation_results.xlsx'")
