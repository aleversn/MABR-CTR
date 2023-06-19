import pandas as pd
pd.set_option('max_row', 200)
root_path = '/project/data/lastfm-dataset-1K/'
merge_path = root_path + 'filter.csv'

df2 = pd.read_csv(merge_path, header=0)
# print(df['skip'].value_counts())
# df = pd.read_csv(root_path + 'track_list.csv')
# # print(df.columns.values)
# track_list = df['Unnamed: 0'].values.tolist()
# dict_track = dict.fromkeys(track_list)
# i = 1
# for item in track_list:
#     dict_track[item] = i
#     i += 1
# print(i)
# df2 = pd.read_csv(merge_path, header=0)
# df = df2['artistname'].value_counts()
# artist_list = list(df.index)
# # print(artist_list)
# dict_artist = dict.fromkeys(artist_list)
# i = 1
# for item in artist_list:
#     dict_artist[item] = i
#     i += 1
# print(i)
# df2['artist_id'] = df2['artistname'].map(dict_artist)

# print(df2[df2['artistname'].isna()])
# df2.to_csv(merge_path, index=False)

def concat_rows(group):
    return pd.Series({
        'userid': group['userid'].iloc[0],
        'concat': ','.join(group['concat'])
    })
# print(df[df['trackid'].isna()])



# df = pd.read_csv(root_path + 'result.csv')
# df['dwell1'] = df['dwell']*1000/df['duration_ms']
#
#
# def judge_dwell(dwell):
#     if dwell < 0.3:
#         return 0
#     elif dwell < 0.6:
#         return 1
#     else:
#         return 2
#
#
# def concat(group):
#     return group['artistid'].astype(str) + '+' + group['trackname'] + '+' + group['skip'].astype(str) \
#            + '+' + group['timestamp'].astype(str)
#
#
# df['skip'] = df['dwell1'].apply(judge_dwell)
# # 按照userid分组，对每个分组进行拼接操作，并将结果保存到新的一列中
# df2['concat'] = df2['artist_id'].astype(str) + '+' + df2['track_name'].astype(str) + '+' + \
#                df2['skip'].astype(str) + '+' + df2['timestamp'].astype(str)
result = df2[["userid", "concat"]]
# # print(result.head())
# df2.to_csv(merge_path, index=False)
# result.to_csv(root_path + 'result.csv', index=False)
# track_path = root_path + 'track_info.csv'
# valid_path = root_path + 'valid6.csv'
# df2 = pd.read_csv(track_path, header=0)
# df1 = pd.read_csv(valid_path, header=0)
# print(df1.shape[0])
# df2.rename(columns={'trackid':'Spotify_trackid'}, inplace=True)
# df2 = df2.drop(labels='id', axis=1)
# merge_df = pd.merge(left=df1, right=df2, on='trackname', how="left", sort=False)
# df3 = merge_df[merge_df['valence'].isna()]
# print(df3.head())
# print('num of na is {}'.format(df3.shape[0]))
# df4 = merge_df[~merge_df['valence'].isna()]
# print(df4.head())
# print('num of not_na is {}'.format(df4.shape[0]))
# df4.to_csv(root_path + 'merge.csv', index=False)
# 按照userid分组，并将concat列拼接为一行
result = result.groupby('userid').apply(concat_rows).reset_index(drop=True)
#
# # 输出拼接结果
print(result)
result['concat'].to_csv(root_path + './valid', index=False, header=False)