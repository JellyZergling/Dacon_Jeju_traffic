import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os

color = sns.color_palette()
sns.set_style('darkgrid')


# CSV to Parquet
def csv_to_parquet(name):
    if not os.path.exists(f'./Data/{name}.parquet'):
        data = pd.read_csv(f'./Data/{name}.csv')
        data.to_parquet(f'./Data/{name}.parquet')
        del data
        gc.collect()


csv_to_parquet('train')
csv_to_parquet('test')

df_train = pd.read_parquet('./Data/train.parquet')
df_test = pd.read_parquet('./Data/test.parquet')

print(df_train.head())
print("-------------------------------")
print(df_test.head())

print(df_train.shape, df_test.shape)

print(df_train.isnull().sum())
print("-------------------------------")
print(df_test.isnull().sum())

print(df_train.info())
print("-------------------------------")
print(df_test.info())

pd.set_option('display.max_columns', None)
print(df_train.head())

change_int32 = ['base_date', 'base_hour', 'lane_count', 'road_rating', 'multi_linked', 'connect_code', 'road_type']
change_float32 = ['maximum_speed_limit', 'vehicle_restricted', 'weight_restricted', 'height_restricted', 'target']

for i in change_int32:
    df_train[i] = df_train[i].astype('int32')
for i in change_float32:
    df_train[i] = df_train[i].astype('float32')

print(df_train.info())

non_show = ['base_date', 'base_hour']
for i in df_train:
    if df_train[i].dtype == 'object':
        non_show.append(i)

plt.figure(figsize=(15, 8))
sns.heatmap(df_train.drop(non_show, axis=1).corr(), annot=True)
plt.show()

# sns.histplot(df_train['maximum_speed_limit'])