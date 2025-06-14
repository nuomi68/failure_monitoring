import pandas as pd
import os
from glob import glob

data_folder = './'

# 合并 PI 点数据中每天12:00的数据
pi_all = pd.DataFrame()
pi_files = glob(os.path.join(data_folder, 'PI点数据*.xlsx'))

for file in pi_files:
    df = pd.read_excel(file, skiprows=1)  # 只跳过第一行，第二行为表头
    df.columns = ['PI时间', '值']
    df['PI时间'] = pd.to_datetime(df['PI时间'])

    # 提取每天 12:00 的数据
    df_12 = df[df['PI时间'].dt.time == pd.to_datetime('12:00:00').time()].copy()
    df_12['日期'] = pd.to_datetime(df_12['PI时间'].dt.date)  # ✅ 强制转成 datetime64[ns]
    pi_all = pd.concat([pi_all, df_12[['日期', 'PI']]])

# 去除重复日期
pi_all = pi_all.drop_duplicates(subset='日期')

# 处理主表
target_file = "./20230510-20240924.xlsx"
df = pd.read_excel(target_file, sheet_name="数据单")

# 提取时间字符串中的日期部分
df["日期"] = pd.to_datetime(df['TIME'], format='%Y年%m月%d日%H%M', errors='coerce')
# 去除时间部分，仅保留日期（仍为 datetime 类型）
df["日期"] = df["日期"].dt.normalize()
# 合并
merged = pd.merge(df, pi_all, on='日期', how='left')
merged.drop(columns=['日期'], inplace=True)

# 保存新文件
output_file = target_file.replace('.xlsx', '_merged.xlsx')
merged.to_excel(output_file, index=False)
print(f'已处理：{output_file}')
