
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
import os



# PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\model compare\\model compare.csv"
#
# table = pd.read_csv(PATH)
# table = table.transpose()
#
# print(table)
#
#
#
# def filter_datatable_by_model_name(df, model_name):
#     new_df = df.copy()
#     return new_df[new_df['model_name'].str.contains(model_name, na=True)]
#
# clip_data = filter_datatable_by_model_name(table, "clip")
# vgg_data = filter_datatable_by_model_name(table, "vgg")
# resnet_data = filter_datatable_by_model_name(table, "resnet")
#
# print(clip_data)


# stab_svm_mean_and_std = {"clip":(0.887440082446553,0.00877192982456148), "vgg16":(0.878537532355478, 0.0127534512510785), "resnet50": (0.882752133064902,0.0127534512510785)}
# stab_ballpark = {"clip":0.891477327197775, "vgg16":0.897744703288275, "resnet50": 0.917541462946985}
#
#
#
# dine_svm_mean_and_std = {"clip":(0.860603953051322, 0.0121309878915967), "vgg16":(0.859297506969406, 0.00894908052946239), "resnet50": (0.845963791897666, 0.00646979244652924)}
# dine_ballpark = {"clip":0.854873180281967, "vgg16":0.900480153999278, "resnet50": 0.890589419112098}
#
#
# buy_svm_mean_and_std = {"clip":(0.9291799014, 0.0125450521482701), "vgg16":(0.893978624866667, 0.00829665234663986), "resnet50": (0.853729667066667, 0.0112978795075245)} # svms mean and std values
# buy_ballpark = {"clip":0.940407305, "vgg16":0.911996537, "resnet50": 0.923892621} #


stab_svm_mean_and_std = {"clip":(0.7829219958,0.010358494), "vgg16":(0.7257068328, 0.0123614235), "resnet50": (0.7351997556,0.00838141450000002)}
stab_ballpark = {"clip":0.785714286, "vgg16":0.765667575, "resnet50":0.773841962}



dine_svm_mean_and_std = {"clip":(0.410031831873557, 0.026085106998728), "vgg16":(0.41275512688583, 0.0275788494416517), "resnet50": (0.399605383409821, 0.0200453749170284)}
dine_ballpark = {"clip":0.420362173231741, "vgg16":0.562793823561052, "resnet50": 0.529251231456553}


buy_svm_mean_and_std = {"clip":(0.3417864514, 0.0270288785125796), "vgg16":(0.282330806066667, 0.00987244830199303), "resnet50": (0.2422452172, 0.0147547312072903)} # svms mean and std values
buy_ballpark = {"clip":0.336619718, "vgg16":0.295165394, "resnet50": 0.323287671} #


labels = ["clip", "vgg16", "resnet50"]

def get_problem_values(ballpark_dict, mean_std_dict):

    values = {"diff":[], "std":[]}
    for label in labels:
        mean, std = mean_std_dict[label]
        ballpark = ballpark_dict[label]
        diff = ballpark-mean
        # min, max = ballpark-min, ballpark-max
        values["diff"].append(diff)
        values["std"].append(std)
        # values["max"].append(max)
    print(values)
    return values


buy = get_problem_values(buy_ballpark, buy_svm_mean_and_std)
dine = get_problem_values(dine_ballpark, dine_svm_mean_and_std)
stab = get_problem_values(stab_ballpark, stab_svm_mean_and_std)
x = np.arange(len(labels))
width = 0.2



fig, ax = plt.subplots()
rects1 = ax.bar(x - width, buy["diff"], width, label='shop', align='edge', yerr=buy["std"],  color="c", capsize=2)
# ax.vlines((x - width/2), buy["max"], buy["min"], color='k')
rects2 = ax.bar(x , dine["diff"], width, label='dine', align='edge', yerr=dine["std"],  color="y", capsize=2)
# ax.vlines((x + width/2), dine["max"], dine["min"], color='k')
rects3 = ax.bar(x + width, stab["diff"], width, label='stab', align='edge', yerr=stab["std"],  color="m", capsize=2)
# ax.vlines((x + width*1.5), stab["max"], stab["min"], color='k')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('pr auc difference between ballpark and psvm average')
# ax.set_title("\n".join(wrap('auc differences between ballpark and the average of polar svm with different bounds')))
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)


fig.tight_layout()
plt.savefig(os.path.join("C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\model compare", "models_compare_pr_auc"), bbox_inches='tight')
plt.show()




men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]
