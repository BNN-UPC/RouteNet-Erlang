############################
########### FIFO ###########
############################
from datanetAPI import DatanetAPI
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# df = pd.read_csv('./original_dataframes/eval/out_gbn_del.txt')
df = pd.read_feather('./original_dataframes/df_with_traffic')
df['delay_rel_error'] = (df["true_delay"] - df["delay"]) / df["true_delay"]
print(np.mean(np.abs(df['delay_rel_error'])) * 100)

df['jitter_rel_error'] = (df["true_jitter"] - df["jitter"]) / df["true_jitter"]
print(np.mean(np.abs(df['jitter_rel_error'])) * 100)

# qt_df = pd.read_csv('./original_dataframes/eval/out_gbn_baseline.txt')
qt_df = pd.read_feather('./original_dataframes/qt_df_with_traffic')
qt_df['delay_rel_error'] = (qt_df["true_delay"] - qt_df["delay"]) / qt_df["true_delay"]

DROP = 0.55
qt_df['abs_jitter'] = np.abs(qt_df['jitter_rel_error'])
qt_df_jitter = qt_df.nsmallest(int(len(qt_df.index) * DROP), 'abs_jitter')

print(np.mean(np.abs(qt_df['delay_rel_error'])) * 100)

qt_df['jitter_rel_error'] = (qt_df["true_jitter"] - qt_df["jitter"]) / qt_df["true_jitter"]
print(np.mean(np.abs(qt_df['jitter_rel_error'])) * 100)

"""api = DatanetAPI("./data/gbnbw", shuffle=True)
it = iter(api)
traffic = []
delay = []
jitter = []
df_with_traffic = pd.DataFrame()
qt_df_with_traffic = pd.DataFrame()
proc_samples = 0
for sample in it:
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()
    for src in range(len(T)):
        for dst in range(len(T)):
            if src != dst:
                traffic = T[src, dst]['AggInfo']['AvgBw']
                delay = P[src, dst]['AggInfo']['AvgDelay']
                jitter = P[src, dst]['AggInfo']['Jitter']
                row = df[df["true_delay"] == delay]
                row = row[row['true_jitter'] == jitter]
                if not row.empty:
                    row['traffic'] = traffic
                    df_with_traffic = df_with_traffic.append(row, ignore_index=True)

                row = qt_df[qt_df["true_delay"] == delay]
                row = row[row['true_jitter'] == jitter]
                if not row.empty:
                    row['traffic'] = traffic
                    qt_df_with_traffic = qt_df_with_traffic.append(row, ignore_index=True)
    print(proc_samples)
    print("df_with_traffic: {}".format(len(df_with_traffic.index)))
    print("qt_df_with_traffic: {}".format(len(qt_df_with_traffic.index)))
    proc_samples += 1
    if proc_samples == 30000:
        break


qt_df_with_traffic.to_feather('./original_dataframes/qt_df_with_traffic')
df_with_traffic.to_feather('./original_dataframes/df_with_traffic')
asdfs"""
plt.rcParams.update({
    "font.size": 16,
    "text.usetex": True})

splitted_qt = np.array_split(qt_df.sort_values('traffic'), 3)
splitter_qt_jitter = np.array_split(qt_df_jitter.sort_values('traffic'), 3)
splitted = np.array_split(df.sort_values('traffic'), 3)

for i in range(len(splitted)):
    df = splitted[i]
    qt_df = splitted_qt[i]
    qt_df_jitter = splitter_qt_jitter[i]
    d_rel_error = list(df['delay_rel_error'])
    d_rel_error.append(0.99)
    plt.hist(d_rel_error,
             cumulative=True,
             histtype='step',
             density=1,
             bins=10000000,
             lw=2,
             color='black',
             zorder=2)
    j_rel_error = list(df['jitter_rel_error'])
    j_rel_error.append(0.99)
    plt.hist(j_rel_error,
             cumulative=True,
             histtype='step',
             density=1,
             bins=10000000,
             lw=2,
             color='black',
             linestyle=(0, (2, 2)),
             zorder=2)
    qt_d_rel_error = list(qt_df['delay_rel_error'])
    qt_d_rel_error.append(0.99)
    plt.hist(qt_d_rel_error,
             cumulative=True,
             histtype='step',
             density=1,
             bins=10000000,
             lw=2,
             color='silver',
             zorder=2)
    qt_j_rel_error = list(qt_df_jitter['jitter_rel_error'])
    qt_j_rel_error.append(0.99)
    plt.hist(qt_j_rel_error,
             cumulative=True,
             histtype='step',
             density=1,
             bins=10000000,
             lw=2,
             color='silver',
             linestyle=(0, (2, 2)),
             zorder=2)
    plt.xlim(-0.75, 0.75)
    plt.legend(('GNN Delay', 'GNN Jitter', 'QT Delay', 'QT Jitter'), fontsize=18, loc='lower right')
    x = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    my_xticks = [r'60\%', r'40\%', r'20\%', r'0\%', r'20\%', r'40\%', r'60\%']
    plt.xticks(x, my_xticks)
    my_yticks = [r'0.0', r'0.2', r'0.4', r'0.6', r'0.8', r'1']
    x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks(x, my_yticks)
    plt.grid(zorder=1)
    plt.ylabel("CDF")
    plt.xlabel(r"Relative Error [$(y - \hat{y})/y \times 100$]")
    plt.tight_layout()
    plt.savefig('./figures/fifo_{}_gnn_{}_{}_qt_{}_{}.pdf'.format(i,
                                                                  round(np.mean(np.abs(df['delay_rel_error'])), 3),
                                                                  round(np.mean(np.abs(df['jitter_rel_error'])), 3),
                                                                  round(np.mean(np.abs(qt_df['delay_rel_error'])), 3),
                                                                  round(
                                                                      np.mean(np.abs(qt_df_jitter['jitter_rel_error'])),
                                                                      3)))
    plt.show()
    plt.close()

############################
######## SCHEDULING ########
############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_feather('scheduling_dataframe')
pred_delay = np.load("./Scheduling/Delay/predictions.npy")
pred_jitter = np.load("./Scheduling/Jitter/predictions.npy")
pred_losses = np.load("./Scheduling/Jitter/predictions.npy")
df["pred_delay"] = np.squeeze(pred_delay)
df["pred_jitter"] = np.squeeze(pred_jitter)
df.to_feather("scheduling_dataframe_with_pred")

df['delay_rel_error'] = (df["delay"] - df["pred_delay"]) / df["delay"]
print(np.mean(np.abs(df['delay_rel_error'])) * 100)

df['jitter_rel_error'] = (df["jitter"] - df["pred_jitter"]) / df["jitter"]
print(np.mean(np.abs(df['jitter_rel_error'])) * 100)

plt.rcParams.update({
    "font.size": 16,
    "text.usetex": True})

splitted = np.array_split(df.sort_values('traffic'), 3)

qt_df = pd.read_csv('./original_dataframes/results_gbn-sp-wfq.csv')
qt_df = qt_df[qt_df['error'] == False]
qt_df = qt_df[qt_df['sim_jitter'] > 0]
qt_df = qt_df[qt_df['qt_jitter'] > 0]
qt_df['delay_rel_error'] = (qt_df["sim_delay"] - qt_df["qt_delay"]) / qt_df["sim_delay"]
qt_df['jitter_rel_error'] = (qt_df["sim_jitter"] - qt_df["qt_jitter"]) / qt_df["sim_jitter"]
qt_df['delay_abs_rel_error'] = np.abs(qt_df['delay_rel_error'])
qt_df['jitter_abs_rel_error'] = np.abs(qt_df['jitter_rel_error'])

DROP = 0.99
qt_df = qt_df.nsmallest(int(len(qt_df.index) * DROP), 'delay_abs_rel_error')
qt_df = qt_df.nsmallest(int(len(qt_df.index) * DROP), 'jitter_abs_rel_error')
print(np.mean(np.abs(qt_df['delay_rel_error'])) * 100)
print(np.mean(np.abs(qt_df['jitter_rel_error'])) * 100)

splitted_qt = np.array_split(qt_df.sort_values('avg_bw'), 3)

qt_df_losses = qt_df[qt_df['sim_loss'] > 0]
qt_df_losses['loss_rel_error'] = (qt_df["sim_loss"] - qt_df["qt_loss"]) / qt_df["sim_loss"]

d_rel_error = list(df['delay_rel_error'])
d_rel_error.append(2)
d_rel_error.append(-2)
plt.hist(d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         zorder=2)
qt_d_rel_error = list(qt_df['delay_rel_error'])
qt_d_rel_error.append(2)
qt_d_rel_error.append(-2)
plt.hist(qt_d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         zorder=2)

plt.xlim(-1.25, 1.25)
plt.legend(('GNN', 'QT'), fontsize=18, loc='upper left')
x = np.array([-1, -0.5, 0.0, 0.5, 1])
my_xticks = [r'-100\%', r'-50\%', r'0\%', r'50\%', r'100\%']
plt.xticks(x, my_xticks)
my_yticks = [r'0.0', r'0.2', r'0.4', r'0.6', r'0.8', r'1']
x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks(x, my_yticks)
plt.grid(zorder=1)
plt.ylabel("CDF")
plt.xlabel(r"Relative Error [$(y - \hat{y})/y \times 100$]")
props = dict(boxstyle='round', facecolor='white', alpha=1)
text = '\n'.join(('GNN MAPE: {}\%'.format(np.round(np.mean(np.abs(df['delay_rel_error'])) * 100,2)), 'QT MAPE: {}\%'.format(np.round(np.mean(np.abs(qt_df['delay_rel_error'])) * 100,2))))
plt.text(0.3, 0.05, text, bbox=props)
plt.tight_layout()
plt.show()
plt.close()

qt_j_rel_error = list(qt_df['jitter_rel_error'])
qt_j_rel_error.append(2)
qt_j_rel_error.append(-2)
plt.hist(qt_j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         linestyle=(0, (2, 2)),
         zorder=2)
j_rel_error = list(df['jitter_rel_error'])
j_rel_error.append(2)
j_rel_error.append(-2)
plt.hist(j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         linestyle=(0, (2, 2)),
         zorder=2)


splitted_qt_losses = np.array_split(qt_df_losses.sort_values('avg_bw'), 3)

for df in splitted_qt_losses:
    print(np.mean(np.abs(df['loss_rel_error'])) * 100)

for i in range(len(splitted)):
    df = splitted[i]
    qt_df = splitted_qt[i]

    plt.hist(df['delay_rel_error'],
             cumulative=True,
             histtype='step',
             density=1,
             bins=10000000,
             lw=2,
             color='black',
             zorder=2)
    plt.hist(df['jitter_rel_error'],
             cumulative=True,
             histtype='step',
             density=1,
             bins=10000000,
             lw=2,
             color='black',
             linestyle=(0, (2, 2)),
             zorder=2)
    qt_d_rel_error = list(qt_df['delay_rel_error'])
    qt_d_rel_error.append(0.99)
    plt.hist(qt_d_rel_error,
             cumulative=True,
             histtype='step',
             density=1,
             bins=10000000,
             lw=2,
             color='silver',
             zorder=2)
    qt_j_rel_error = list(qt_df['jitter_rel_error'])
    qt_j_rel_error.append(0.99)
    plt.hist(qt_j_rel_error,
             cumulative=True,
             histtype='step',
             density=1,
             bins=10000000,
             lw=2,
             color='silver',
             linestyle=(0, (2, 2)),
             zorder=2)
    plt.xlim(-0.75, 0.75)
    plt.legend(('GNN Delay', 'GNN Jitter', 'QT Delay', 'QT Jitter'), fontsize=18, loc='lower right')
    x = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    my_xticks = [r'60\%', r'40\%', r'20\%', r'0\%', r'20\%', r'40\%', r'60\%']
    plt.xticks(x, my_xticks)
    my_yticks = [r'0.0', r'0.2', r'0.4', r'0.6', r'0.8', r'1']
    x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks(x, my_yticks)
    plt.grid(zorder=1)
    plt.ylabel("CDF")
    plt.xlabel(r"Relative Error [$(y - \hat{y})/y \times 100$]")
    plt.tight_layout()
    plt.savefig('./figures/scheduling_{}_gnn_{}_{}_qt_{}_{}.pdf'.format(i,
                                                                        round(np.mean(np.abs(df['delay_rel_error'])),
                                                                              3),
                                                                        round(np.mean(np.abs(df['jitter_rel_error'])),
                                                                              3),
                                                                        round(np.mean(np.abs(qt_df['delay_rel_error'])),
                                                                              3),
                                                                        round(
                                                                            np.mean(np.abs(qt_df['jitter_rel_error'])),
                                                                            3)))
    plt.show()
    plt.close()

############################
####### DETERMINISTIC ######
############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_feather('./original_dataframes/deterministic_dataframe')
pred_delay = np.load("./original_dataframes/deterministic_delay_predictions.npy")
pred_jitter = np.load("./original_dataframes/deterministic_jitter_predictions.npy")
df["pred_delay"] = np.squeeze(pred_delay)
df["pred_jitter"] = np.squeeze(pred_jitter)

df['delay_rel_error'] = (df["delay"] - df["pred_delay"]) / df["delay"]
print(np.mean(np.abs(df['delay_rel_error'])) * 100)

df['jitter_rel_error'] = (df["jitter"] - df["pred_jitter"]) / df["jitter"]
print(np.mean(np.abs(df['jitter_rel_error'])) * 100)

qt_df = pd.read_csv('./original_dataframes/results_gbn-det.csv')
qt_df = qt_df[qt_df['error'] == False]
qt_df = qt_df[qt_df['sim_jitter'] != 0]
qt_df = qt_df[qt_df['qt_jitter'] != 0]
qt_df['delay_rel_error'] = (qt_df["sim_delay"] - qt_df["qt_delay"]) / qt_df["sim_delay"]
print(np.mean(np.abs(qt_df['delay_rel_error'])) * 100)

qt_df['jitter_rel_error'] = (qt_df["sim_jitter"] - qt_df["qt_jitter"]) / qt_df[
    "sim_jitter"]
print(np.mean(np.abs(qt_df['jitter_rel_error'])) * 100)

plt.rcParams.update({
    "font.size": 16,
    "text.usetex": True})

d_rel_error = list(df['delay_rel_error'])
d_rel_error.append(2)
d_rel_error.append(-2)
plt.hist(d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         zorder=2)
j_rel_error = list(df['jitter_rel_error'])
j_rel_error.append(2)
j_rel_error.append(-2)
plt.hist(j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         linestyle=(0, (2, 2)),
         zorder=2)
qt_d_rel_error = list(qt_df['delay_rel_error'])
qt_d_rel_error.append(2)
qt_d_rel_error.append(-2)
plt.hist(qt_d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         zorder=2)
qt_j_rel_error = list(qt_df['jitter_rel_error'])
qt_j_rel_error.append(2)
qt_j_rel_error.append(-2)
plt.hist(qt_j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         linestyle=(0, (2, 2)),
         zorder=2)
plt.xlim(-1.25, 1.25)
plt.legend(('GNN Delay', 'GNN Jitter', 'QT Delay', 'QT Jitter'), fontsize=18, loc='upper left')
x = np.array([-1, -0.5, 0.0, 0.5, 1])
my_xticks = [r'-100\%', r'-50\%', r'0\%', r'50\%', r'100\%']
plt.xticks(x, my_xticks)
my_yticks = [r'0.0', r'0.2', r'0.4', r'0.6', r'0.8', r'1']
x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks(x, my_yticks)
plt.grid(zorder=1)
plt.ylabel("CDF")
plt.xlabel(r"Relative Error [$(y - \hat{y})/y \times 100$]")
plt.tight_layout()
plt.savefig('./figures/deterministic_gnn_{}_{}_qt_{}_{}.pdf'.format(
    round(np.mean(np.abs(df['delay_rel_error'])), 3),
    round(np.mean(np.abs(df['jitter_rel_error'])), 3),
    round(np.mean(np.abs(qt_df['delay_rel_error'])), 3),
    round(np.mean(np.abs(qt_df['jitter_rel_error'])), 3)))
plt.show()
plt.close()

############################
########### ONOFF ##########
############################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_feather('./original_dataframes/onoff_dataframe')
pred_delay = np.load("./original_dataframes/onoff_delay_predictions.npy")
pred_jitter = np.load("./original_dataframes/onoff_jitter_predictions.npy")
df["pred_delay"] = np.squeeze(pred_delay)
df["pred_jitter"] = np.squeeze(pred_jitter)

df['delay_rel_error'] = (df["delay"] - df["pred_delay"]) / df["delay"]
print(np.mean(np.abs(df['delay_rel_error'])) * 100)

df['jitter_rel_error'] = (df["jitter"] - df["pred_jitter"]) / df["jitter"]
print(np.mean(np.abs(df['jitter_rel_error'])) * 100)

qt_df = pd.read_csv('./original_dataframes/results_gbn-onoff.csv')
qt_df = qt_df[qt_df['error'] == False]
qt_df = qt_df[qt_df['sim_jitter'] != 0]
qt_df = qt_df[qt_df['qt_jitter'] != 0]
qt_df['delay_rel_error'] = (qt_df["sim_delay"] - qt_df["qt_delay"]) / qt_df["sim_delay"]
print(np.mean(np.abs(qt_df['delay_rel_error'])) * 100)

qt_df['jitter_rel_error'] = (qt_df["sim_jitter"] - qt_df["qt_jitter"]) / qt_df[
    "sim_jitter"]
print(np.mean(np.abs(qt_df['jitter_rel_error'])) * 100)

plt.rcParams.update({
    "font.size": 16,
    "text.usetex": True})

d_rel_error = list(df['delay_rel_error'])
d_rel_error.append(2)
d_rel_error.append(-2)
plt.hist(d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         zorder=2)
j_rel_error = list(df['jitter_rel_error'])
j_rel_error.append(2)
j_rel_error.append(-2)
plt.hist(j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         linestyle=(0, (2, 2)),
         zorder=2)
qt_d_rel_error = list(qt_df['delay_rel_error'])
qt_d_rel_error.append(2)
qt_d_rel_error.append(-2)
plt.hist(qt_d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         zorder=2)
qt_j_rel_error = list(qt_df['jitter_rel_error'])
qt_j_rel_error.append(2)
qt_j_rel_error.append(-2)
plt.hist(qt_j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         linestyle=(0, (2, 2)),
         zorder=2)
plt.xlim(-1.25, 1.25)
plt.legend(('GNN Delay', 'GNN Jitter', 'QT Delay', 'QT Jitter'), fontsize=18, loc='upper left')
x = np.array([-1, -0.5, 0.0, 0.5, 1])
my_xticks = [r'-100\%', r'-50\%', r'0\%', r'50\%', r'100\%']
plt.xticks(x, my_xticks)
my_yticks = [r'0.0', r'0.2', r'0.4', r'0.6', r'0.8', r'1']
x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks(x, my_yticks)
plt.grid(zorder=1)
plt.ylabel("CDF")
plt.xlabel(r"Relative Error [$(y - \hat{y})/y \times 100$]")
plt.tight_layout()
plt.savefig('./figures/onoff_gnn_{}_{}_qt_{}_{}.pdf'.format(
    round(np.mean(np.abs(df['delay_rel_error'])), 3),
    round(np.mean(np.abs(df['jitter_rel_error'])), 3),
    round(np.mean(np.abs(qt_df['delay_rel_error'])), 3),
    round(np.mean(np.abs(qt_df['jitter_rel_error'])), 3)))
plt.show()
plt.close()

############################
############ K1 ############
############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_feather('./original_dataframes/k1_dataframe')
pred_delay = np.load("./original_dataframes/k1_delay_predictions.npy")
pred_jitter = np.load("./original_dataframes/k1_jitter_predictions.npy")
df["pred_delay"] = np.squeeze(pred_delay)
df["pred_jitter"] = np.squeeze(pred_jitter)

df['delay_rel_error'] = (df["delay"] - df["pred_delay"]) / df["delay"]
print(np.mean(np.abs(df['delay_rel_error'])) * 100)

df['jitter_rel_error'] = (df["jitter"] - df["pred_jitter"]) / df["jitter"]
print(np.mean(np.abs(df['jitter_rel_error'])) * 100)

qt_df = pd.read_csv('./original_dataframes/results_gbn_k1.csv')
qt_df = qt_df[qt_df['error'] == False]
qt_df = qt_df[qt_df['sim_jitter'] != 0]
qt_df = qt_df[qt_df['qt_jitter'] != 0]
qt_df['delay_rel_error'] = (qt_df["sim_delay"] - qt_df["qt_delay"]) / qt_df["sim_delay"]
print(np.mean(np.abs(qt_df['delay_rel_error'])) * 100)

qt_df['jitter_rel_error'] = (qt_df["sim_jitter"] - qt_df["qt_jitter"]) / qt_df[
    "sim_jitter"]
print(np.mean(np.abs(qt_df['jitter_rel_error'])) * 100)

plt.rcParams.update({
    "font.size": 16,
    "text.usetex": True})

d_rel_error = list(df['delay_rel_error'])
d_rel_error.append(2)
d_rel_error.append(-2)
plt.hist(d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         zorder=2)
j_rel_error = list(df['jitter_rel_error'])
j_rel_error.append(2)
j_rel_error.append(-2)
plt.hist(j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         linestyle=(0, (2, 2)),
         zorder=2)
qt_d_rel_error = list(qt_df['delay_rel_error'])
qt_d_rel_error.append(2)
qt_d_rel_error.append(-2)
plt.hist(qt_d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         zorder=2)
qt_j_rel_error = list(qt_df['jitter_rel_error'])
qt_j_rel_error.append(2)
qt_j_rel_error.append(-2)
plt.hist(qt_j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         linestyle=(0, (2, 2)),
         zorder=2)
plt.xlim(-1.25, 1.25)
plt.legend(('GNN Delay', 'GNN Jitter', 'QT Delay', 'QT Jitter'), fontsize=18, loc='upper left')
x = np.array([-1, -0.5, 0.0, 0.5, 1])
my_xticks = [r'-100\%', r'-50\%', r'0\%', r'50\%', r'100\%']
plt.xticks(x, my_xticks)
my_yticks = [r'0.0', r'0.2', r'0.4', r'0.6', r'0.8', r'1']
x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks(x, my_yticks)
plt.grid(zorder=1)
plt.ylabel("CDF")
plt.xlabel(r"Relative Error [$(y - \hat{y})/y \times 100$]")
plt.tight_layout()
"""plt.savefig('./figures/k1_gnn_{}_{}_qt_{}_{}.pdf'.format(
    round(np.mean(np.abs(df['delay_rel_error'])), 3),
    round(np.mean(np.abs(df['jitter_rel_error'])), 3),
    round(np.mean(np.abs(qt_df['delay_rel_error'])), 3),
    round(np.mean(np.abs(qt_df['jitter_rel_error'])), 3)))"""
plt.show()
plt.close()

############################
############ K2 ############
############################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_feather('./original_dataframes/k2_dataframe')
pred_delay = np.load("./original_dataframes/k2_delay_predictions.npy")
pred_jitter = np.load("./original_dataframes/k2_jitter_predictions.npy")
df["pred_delay"] = np.squeeze(pred_delay)
df["pred_jitter"] = np.squeeze(pred_jitter)

df['delay_rel_error'] = (df["delay"] - df["pred_delay"]) / df["delay"]
print(np.mean(np.abs(df['delay_rel_error'])) * 100)

df['jitter_rel_error'] = (df["jitter"] - df["pred_jitter"]) / df["jitter"]
print(np.mean(np.abs(df['jitter_rel_error'])) * 100)

qt_df = pd.read_csv('./original_dataframes/results_gbn-k2.csv')
qt_df = qt_df[qt_df['error'] == False]
qt_df = qt_df[qt_df['sim_jitter'] != 0]
qt_df = qt_df[qt_df['qt_jitter'] != 0]
qt_df['delay_rel_error'] = (qt_df["sim_delay"] - qt_df["qt_delay"]) / qt_df["sim_delay"]
print(np.mean(np.abs(qt_df['delay_rel_error'])) * 100)

qt_df['jitter_rel_error'] = (qt_df["sim_jitter"] - qt_df["qt_jitter"]) / qt_df[
    "sim_jitter"]
print(np.mean(np.abs(qt_df['jitter_rel_error'])) * 100)

plt.rcParams.update({
    "font.size": 16,
    "text.usetex": True})

d_rel_error = list(df['delay_rel_error'])
d_rel_error.append(2)
d_rel_error.append(-2)
plt.hist(d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         zorder=2)
j_rel_error = list(df['jitter_rel_error'])
j_rel_error.append(2)
j_rel_error.append(-2)
plt.hist(j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         linestyle=(0, (2, 2)),
         zorder=2)
qt_d_rel_error = list(qt_df['delay_rel_error'])
qt_d_rel_error.append(2)
qt_d_rel_error.append(-2)
plt.hist(qt_d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         zorder=2)
qt_j_rel_error = list(qt_df['jitter_rel_error'])
qt_j_rel_error.append(2)
qt_j_rel_error.append(-2)
plt.hist(qt_j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         linestyle=(0, (2, 2)),
         zorder=2)
plt.xlim(-1.25, 1.25)
plt.legend(('GNN Delay', 'GNN Jitter', 'QT Delay', 'QT Jitter'), fontsize=18, loc='upper left')
x = np.array([-1, -0.5, 0.0, 0.5, 1])
my_xticks = [r'-100\%', r'-50\%', r'0\%', r'50\%', r'100\%']
plt.xticks(x, my_xticks)
my_yticks = [r'0.0', r'0.2', r'0.4', r'0.6', r'0.8', r'1']
x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks(x, my_yticks)
plt.grid(zorder=1)
plt.ylabel("CDF")
plt.xlabel(r"Relative Error [$(y - \hat{y})/y \times 100$]")
plt.tight_layout()
plt.savefig('./figures/k2_gnn_{}_{}_qt_{}_{}.pdf'.format(
    round(np.mean(np.abs(df['delay_rel_error'])), 3),
    round(np.mean(np.abs(df['jitter_rel_error'])), 3),
    round(np.mean(np.abs(qt_df['delay_rel_error'])), 3),
    round(np.mean(np.abs(qt_df['jitter_rel_error'])), 3)))
plt.show()
plt.close()

############################
########## MIXED ###########
############################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_feather('./original_dataframes/mixed_dataframe')
pred_delay = np.load("./original_dataframes/mixed_delay_predictions.npy")
pred_jitter = np.load("./original_dataframes/mixed_jitter_predictions.npy")
df["pred_delay"] = np.squeeze(pred_delay)
df["pred_jitter"] = np.squeeze(pred_jitter)

df['delay_rel_error'] = (df["delay"] - df["pred_delay"]) / df["delay"]
print(np.mean(np.abs(df['delay_rel_error'])) * 100)

df['jitter_rel_error'] = (df["jitter"] - df["pred_jitter"]) / df["jitter"]
print(np.mean(np.abs(df['jitter_rel_error'])) * 100)

qt_df = pd.read_csv('./original_dataframes/results_var-gbn.csv')
qt_df = qt_df[qt_df['error'] == False]
qt_df = qt_df[qt_df['sim_jitter'] != 0]
qt_df = qt_df[qt_df['qt_jitter'] != 0]
qt_df['delay_rel_error'] = (qt_df["sim_delay"] - qt_df["qt_delay"]) / qt_df["sim_delay"]
print(np.mean(np.abs(qt_df['delay_rel_error'])) * 100)

qt_df['jitter_rel_error'] = (qt_df["sim_jitter"] - qt_df["qt_jitter"]) / qt_df[
    "sim_jitter"]
print(np.mean(np.abs(qt_df['jitter_rel_error'])) * 100)

plt.rcParams.update({
    "font.size": 16,
    "text.usetex": True})

d_rel_error = list(df['delay_rel_error'])
d_rel_error.append(2)
d_rel_error.append(-2)
plt.hist(d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         zorder=2)
j_rel_error = list(df['jitter_rel_error'])
j_rel_error.append(2)
j_rel_error.append(-2)
plt.hist(j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='black',
         linestyle=(0, (2, 2)),
         zorder=2)
qt_d_rel_error = list(qt_df['delay_rel_error'])
qt_d_rel_error.append(2)
qt_d_rel_error.append(-2)
plt.hist(qt_d_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         zorder=2)
qt_j_rel_error = list(qt_df['jitter_rel_error'])
qt_j_rel_error.append(2)
qt_j_rel_error.append(-2)
plt.hist(qt_j_rel_error,
         cumulative=True,
         histtype='step',
         density=1,
         bins=10000000,
         lw=2,
         color='silver',
         linestyle=(0, (2, 2)),
         zorder=2)
plt.xlim(-1.25, 1.25)
plt.legend(('GNN Delay', 'GNN Jitter', 'QT Delay', 'QT Jitter'), fontsize=18, loc='upper left')
x = np.array([-1, -0.5, 0.0, 0.5, 1])
my_xticks = [r'-100\%', r'-50\%', r'0\%', r'50\%', r'100\%']
plt.xticks(x, my_xticks)
my_yticks = [r'0.0', r'0.2', r'0.4', r'0.6', r'0.8', r'1']
x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks(x, my_yticks)
plt.grid(zorder=1)
plt.ylabel("CDF")
plt.xlabel(r"Relative Error [$(y - \hat{y})/y \times 100$]")
plt.tight_layout()
"""plt.savefig('./figures/mixed_gnn_{}_{}_qt_{}_{}.pdf'.format(
    round(np.mean(np.abs(df['delay_rel_error'])), 3),
    round(np.mean(np.abs(df['jitter_rel_error'])), 3),
    round(np.mean(np.abs(qt_df['delay_rel_error'])), 3),
    round(np.mean(np.abs(qt_df['jitter_rel_error'])), 3)))"""
plt.show()
plt.close()

############################
########### K1 A ###########
############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 16,
    "text.usetex": True})

qt_df = pd.read_feather('./original_dataframes/results_gbn-k1-5-9')
qt_df = qt_df[qt_df['error'] == False]
qt_df = qt_df[qt_df['sim_jitter'] != 0]
qt_df = qt_df[qt_df['qt_jitter'] != 0]
qt_df['delay_rel_error'] = (qt_df["sim_delay"] - qt_df["qt_delay"]) / qt_df["sim_delay"]
qt_df['jitter_rel_error'] = (qt_df["sim_jitter"] - qt_df["qt_jitter"]) / qt_df[
    "sim_jitter"]
divisions = np.arange(0.5, 0.91, 0.1)
data = []

for i in range(len(divisions) - 1):
    data.append(
        list(np.abs(qt_df[qt_df['a'] > divisions[i]][qt_df['a'] <= divisions[i + 1]]['jitter_rel_error']) * 100))

plt.boxplot(data, showfliers=False)
plt.show()

for elem in data:
    print(np.mean(elem))

####################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

qt_df = pd.read_csv('./original_dataframes/results_gbn-det.csv')
qt_df['delay_rel_error'] = (qt_df["sim_delay"] - qt_df["qt_delay"]) / qt_df["sim_delay"]
qt_df['jitter_rel_error'] = (qt_df["sim_jitter"] - qt_df["qt_jitter"]) / qt_df["sim_jitter"]

print(np.mean(np.abs(qt_df['delay_rel_error'])) * 100)
print(np.mean(np.abs(qt_df['jitter_rel_error'])) * 100)
splitted = np.array_split(qt_df.sort_values('avg_bw'), 4)

data = []
for df in splitted:
    data.append(np.abs(list(df['delay_rel_error'])) * 100)

plt.boxplot(data, showfliers=False)
plt.show()

for elem in data:
    print(np.round(np.mean(elem), 2))

qt_df['avg_bw'].describe()

############################
####### SCALABILITY ########
############################
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "text.usetex": True})

with open('./original_dataframes/scalability_results.pkl', 'rb') as handle:
    data = pickle.load(handle)

keys = list(data.keys())
keys.sort()
divisions = list(range(50, 301, 50))
boxplot_data = [[] for _ in range(len(divisions) - 1)]
for i in range(len(divisions) - 1):
    for key in keys:
        if divisions[i] <= key < divisions[i + 1]:
            boxplot_data[i].extend(np.abs(data[key]) * 100)

for i in range(len(boxplot_data)):
    boxplot_data[i].sort()
    boxplot_data[i] = boxplot_data[i][:int(len(boxplot_data[i]) * 0.8)]

plt.boxplot(boxplot_data, showfliers=False, showmeans=True, meanline=True)
plt.xlabel("Topology Size")
plt.ylabel("Absolute Relative Error")
x = np.array([1, 2, 3, 4, 5])
my_xticks = [r'[50,99]', r'[100,149]', r'[150,199]', r'[200,249]', r'[250,300]']
plt.xticks(x, my_xticks)
x = np.array([0, 5, 10, 15, 20])
my_xticks = [r'0\%', r'5\%', r'10\%', r'15\%', r'20\%']
plt.yticks(x, my_xticks)
plt.grid()
plt.tight_layout()
plt.savefig('./figures/scalability_results.pdf')
plt.show()
plt.close()

############################
########## SPEED ###########
############################
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams.update({
    "font.size": 16,
    "text.usetex": True})

top_size = list(range(10, 81, 5))
qt_time = []
for elem in top_size:
    path_len = np.log(elem)
    G = nx.erdos_renyi_graph(elem, 0.25)
    qt_time.append(np.round(0.000511 * G.number_of_edges(), 5))
gnn_time = [0.049, 0.045, 0.045, 0.045, 0.054, 0.06, 0.059, 0.057, 0.075, 0.09, 0.12, 0.109, 0.13, 0.14, 0.17]
sim_time = [10, 100, 125, 200, 300, 650, 750, 900, 1450, 1580, 1900, 2700, 3800, 5000, 6500]
qt_time = [0.00766, 0.01635, 0.01942, 0.03321, 0.05008, 0.07563, 0.09862, 0.13082, 0.1625, 0.19111, 0.23557, 0.28105,
           0.30558, 0.37558, 0.39807]

plt.plot(top_size, sim_time, linestyle='solid', marker='.', color='black', linewidth=2.5, markersize=20)
plt.plot(top_size, gnn_time, linestyle='dashed', marker='8', color='grey', linewidth=2.5, markersize=10)
plt.plot(top_size, qt_time, linestyle='dashdot', marker='s', color='silver', linewidth=2.5, markersize=10)
plt.legend(('Simulator', 'GNN', 'QT'), fontsize=18, loc='upper left')
plt.grid()
plt.xlabel("Topology size (Number of nodes)")
plt.ylabel("Time (seconds)")
plt.tight_layout()
# plt.savefig('./figures/exec_time.pdf')
plt.yscale('log')
plt.show()
plt.close()

plt.plot(top_size, sim_time, linestyle='solid', marker='.', color='black', linewidth=2.5, markersize=20)
plt.plot(top_size, gnn_time, linestyle='dashed', marker='8', color='grey', linewidth=2.5, markersize=10)
plt.plot(top_size, qt_time, linestyle='dashdot', marker='s', color='silver', linewidth=2.5, markersize=10)
plt.legend(('Simulator', 'GNN', 'QT'), fontsize=18, loc='upper left')
plt.grid()
plt.xlabel("Topology size (Number of nodes)")
plt.ylabel("Time (seconds)")
plt.tight_layout()
plt.ylim(-0.015, 0.45)
plt.savefig('./figures/exec_time_zoomed.pdf')
plt.show()
plt.close()

plt.plot(top_size, np.array(sim_time) / np.array(gnn_time), linestyle='dashed', marker='8', color='grey', linewidth=2.5,
         markersize=10)
plt.plot(top_size, np.array(sim_time) / np.array(qt_time), linestyle='dashdot', marker='s', color='silver',
         linewidth=2.5, markersize=10)
plt.grid()
plt.xlabel("Topology size")
plt.legend(('GNN', 'QT'), fontsize=18, loc='upper left')
plt.tight_layout()
plt.show()
