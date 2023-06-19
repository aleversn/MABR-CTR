import spotipy
import pandas
from spotipy import SpotifyOAuth


def save_list(i, tlist):
    if i == 1:
        df = pandas.DataFrame(tlist)
    else:
        df = pandas.read_csv('./track_id.csv', header=None, names=['trackid'])
        df_list = df['trackid'].values.tolist()
        df_list += tlist
        df = pandas.DataFrame(df_list)
    df.to_csv('./track_id.csv', index=False)
    print('=====================================save it============================')


SPOTIFY_CLIENT_ID = ''
SPOTIFY_CLIENT_SECRET = ''
SPOTIFY_SCOPE = 'user-library-read'
SPOTIFY_USERNAME = ''
SPOTIFY_REDIRECT_URI = 'http://localhost:8888/'
auth_manager = SpotifyOAuth(client_id=SPOTIFY_CLIENT_ID,
                            client_secret=SPOTIFY_CLIENT_SECRET,
                            scope=SPOTIFY_SCOPE,
                            username=SPOTIFY_USERNAME,
                            redirect_uri=SPOTIFY_REDIRECT_URI)

sp = spotipy.Spotify(auth_manager=auth_manager)  # 完成这步后就可以开始使用Api了
# resp = sp.audio_features('2ZULEJnMCy8S54UwCecbJr')
# print(resp)
# for name in namelist:
# track_list = pandas.read_csv('./track_name.csv')
# # print(track_list.head())
# name_list = track_list['Unnamed: 0'].values.tolist()
# id_list = []
# # i = 0
# i = 11300
# for j in range(11299, len(name_list)):
#     print('{} + {}'.format(str(i), name_list[j]))
#     s = name_list[j].find('(')
#     if s != -1 and s != 0:
#         q_name = name_list[j][:s - 1]
#     elif s == -1:
#         q_name = name_list[j]
#     else:
#         s = name_list[j].find('(', s + 1)
#         if s == -1:
#             q_name = name_list[j]
#         else:
#             q_name = name_list[j][:s - 1]
#     q = 'track:' + q_name
#     resp = sp.search(q=q, type='track')
#     sp.audio_features()
#     i += 1
#     print(resp['tracks']['items'])
#     if len(resp['tracks']['items']) > 0:
#         id = resp['tracks']['items'][0]['id']
#         id_list.append(id)
#     else:
#         id_list.append("")
#     if i % 100 == 0:
#         save_list(i/100, id_list)
#         id_list = []
# save_list(i/100, id_list)
# track_list["track_spotify_id"] = id_list
# track_list.to_csv('./track_list.csv', index=False)
df1 = pandas.read_csv('./track_name.csv', names=['trackname', 'counts'])
df2 = pandas.read_csv('./track_id.csv')
df1['trackid'] = df2['trackid'].values.tolist()
print(df1.head())
df1.to_csv('./track_name_id.csv', index=False)
# df = pandas.DataFrame(id_list)
# df.to_csv('./track_id1.csv', index=False)
# df = pandas.read_csv('./track_id.csv', names=['id'])
# id_list = df[~df[id].isna()].values.tolist()
# danceabilty_list, energy_list, key_list, loudness_list, mode_list, speechiness_list, acousticness_list,\
# instrumentalness_list, liveness_list, valence_list, tempo_list, duration_ms_list = [], [], [], [], [], [], [],\
#                                                                                    [], [], [], [], []
# q_list = []
#
# for i in range(len(id_list)):
#     if id_list[i] == 0:
#         danceabilty_list.append('')
#         energy_list.append('')
#         key_list.append('')
#         loudness_list.append('')
#         mode_list.append('')
#         speechiness_list.append('')
#         acousticness_list.append('')
#         instrumentalness_list.append('')
#         liveness_list.append('')
#         valence_list.append('')
#         tempo_list.append('')
#         duration_ms_list.append('')
#     elif id_list[i] == '':
#         danceabilty_list.append('')
#         energy_list.append('')
#         key_list.append('')
#         loudness_list.append('')
#         mode_list.append('')
#         speechiness_list.append('')
#         acousticness_list.append('')
#         instrumentalness_list.append('')
#         liveness_list.append('')
#         valence_list.append('')
#         tempo_list.append('')
#         duration_ms_list.append('')
#         break
#     else:
#         q_list.append(id_list[i])
# resp = sp.audio_features()
# for line in resp:
#     danceabilty_list.append(line['danceability'])
#     energy_list.append(line['energy'])
#     key_list.append(line['key'])
#     loudness_list.append(line['loudness'])
#     mode_list.append(line['mode'])
#     speechiness_list.append(line['speechiness'])
#     acousticness_list.append(line['acousticness'])
#     instrumentalness_list.append(line['instrumentalness'])
#     liveness_list.append(line['liveness'])
#     valence_list.append(line['valence'])
#     tempo_list.append(line['tempo'])
#     duration_ms_list.append(line['duration_ms'])
