from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

data = pd.read_excel('heatmap.xlsx')
head1 = data.iloc[0:4]
head2 = data.iloc[4::]
head1.index = ['itemID', 'behavior_type', 'dwell', 'gap']
head2.index = ['itemID', 'behavior_type', 'dwell', 'gap']
fig = plt.figure(figsize=(15, 6), layout="tight")
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.xaxis.tick_top()
ax1 = sns.heatmap(head1, cmap="OrRd", ax=ax1)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=14)
plt.setp(ax1.get_xticklabels(), fontsize=14)
ax2 = fig.add_subplot(gs[0, 1])
ax2.xaxis.tick_top()
ax2 = sns.heatmap(head2, cmap="OrRd", ax=ax2)
plt.setp(ax2.get_xticklabels(), fontsize=14)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=14)
# fig, ax = plt.subplots(1, 2, figsize=(30, 12))
# ax[0].xaxis.tick_top()
# ax[1].xaxis.tick_top()
# ax[0].xticks(fontsize=16)
# ax[0].yticks(rotation=0)
# sns.heatmap(head1, cmap="OrRd", ax=ax[0])
#
# sns.heatmap(head2, cmap="OrRd", ax=ax[1])
# plt.xticks(fontsize=16)
# plt.yticks(rotation=0, fontsize=16)
plt.savefig("heatmap.jpg")
plt.show()