import pandas as pd
import matplotlib.pyplot as plt
import os



# PATHS = "19$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\explore_0\\regress_results\\evaluations.csv," \
#         "26$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\explore_1\\regress_results\\evaluations.csv," \
#         "32$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\explore_2\\regress_results\\evaluations.csv," \
#         "37$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\explore_3\\regress_results\\evaluations.csv"
#         # "40$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\dine\\manual_constraints\\explore_3_noise\\noise_40\\regress_results\\evaluations.csv"

# GRAPH_FIELD = "precision_recall_auc"
# X_LABEL = "number of constrained classes"
#
# OUTPUT_PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\dine\\manual_constraints\\iter_graph_new_pr_auc"

PATHS = "0.0$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig00_noise100\\regress_results\\evaluations.csv," \
        "0.1$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig01_noise100\\regress_results\\evaluations.csv," \
        "0.2$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig02_noise100\\regress_results\\evaluations.csv," \
        "0.3$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig03_noise100\\regress_results\\evaluations.csv," \
        "0.4$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig04_noise100\\regress_results\\evaluations.csv,"\
        "0.5$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig05_noise100\\regress_results\\evaluations.csv," \
        "0.6$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig06_noise100\\regress_results\\evaluations.csv," \
        "0.8$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig07_noise100\\regress_results\\evaluations.csv," \
        "0.8$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig08_noise100\\regress_results\\evaluations.csv," \
        "0.8$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig09_noise100\\regress_results\\evaluations.csv," \
        "1.0$C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\orig10_noise100\\regress_results\\evaluations.csv"

GRAPH_FIELD = "auc"
X_LABEL = "the size of the gt data"

OUTPUT_PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\stab\\manual_constraints\\noise_explore\\iter_graph\\noisy_data_gt_addition_auc"



def merge_data(iter2path, graph_field):
    merged_table = None
    for itername in iter2path:
        filepath = iter2path[itername]

        table = pd.read_csv(filepath)[["model_name", graph_field]]
        table = table.rename(columns={graph_field: itername})
        if merged_table is None:
            merged_table = table
        else:
            merged_table = merged_table.merge(table, how='left', on="model_name")
    return merged_table

def add_mean_graph(exclude_key_words, data_table, model_name):
    filtered_data_table = data_table.copy()
    for key in exclude_key_words:
        filtered_data_table = filtered_data_table[~filtered_data_table['model_name'].str.contains(key, na=True)]
    filtered_data_table = filtered_data_table.drop('model_name', axis=1)
    filtered_data_table = filtered_data_table.mean(axis=0).to_frame().transpose()
    filtered_data_table['model_name'] = pd.Series(["{}_svm_mean".format(model_name)], index=filtered_data_table.index)
    # data_table.merge(filtered_data_table, how='left')
    filtered_data_table = data_table.merge(filtered_data_table, how="outer")
    return filtered_data_table

def add_std_graph(exclude_key_words, data_table):
    filtered_data_table = data_table.copy()
    for key in exclude_key_words:
        filtered_data_table = filtered_data_table[~filtered_data_table['model_name'].str.contains(key, na=True)]
    filtered_data_table = filtered_data_table.drop('model_name', axis=1)
    # mean_data_table = filtered_data_table.copy().mean(axis=0).to_frame().transpose()
    std_data_table = filtered_data_table.std(axis=0).to_frame().transpose()
    # print(mean_data_table)
    # print(std_data_table)
    # print(mean_data_table - std_data_table)


    std_data_table['model_name'] = pd.Series(["error"], index=std_data_table.index)
    # data_table.merge(filtered_data_table, how='left')
    std_data_table = data_table.merge(std_data_table, how="outer")
    return std_data_table

def add_max_graph(exclude_key_words, data_table):
    filtered_data_table = data_table.copy()
    for key in exclude_key_words:
        filtered_data_table = filtered_data_table[~filtered_data_table['model_name'].str.contains(key, na=True)]
    filtered_data_table = filtered_data_table.drop('model_name', axis=1)
    filtered_data_table = filtered_data_table.max(axis=0).to_frame().transpose()
    filtered_data_table['model_name'] = pd.Series(["svm_std"], index=filtered_data_table.index)
    # data_table.merge(filtered_data_table, how='left')
    filtered_data_table = data_table.merge(filtered_data_table, how="outer")
    return filtered_data_table

def add_min_graph(exclude_key_words, data_table):
    filtered_data_table = data_table.copy()
    for key in exclude_key_words:
        filtered_data_table = filtered_data_table[~filtered_data_table['model_name'].str.contains(key, na=True)]
    filtered_data_table = filtered_data_table.drop('model_name', axis=1)
    filtered_data_table = filtered_data_table.min(axis=0).to_frame().transpose()
    filtered_data_table['model_name'] = pd.Series(["svm_min"], index=filtered_data_table.index)
    # data_table.merge(filtered_data_table, how='left')
    filtered_data_table = data_table.merge(filtered_data_table, how="outer")
    return filtered_data_table

def filter_datatable_by_model_name(df, model_name):
    new_df = df.copy()
    return new_df[new_df['model_name'].str.contains(model_name, na=True)]





def main():
    paths = PATHS.split(",")
    iter2path = dict()
    for path in paths:
        param = path.split("$")
        iter2path[param[0]] = param[1]
    merged_table = merge_data(iter2path, GRAPH_FIELD)

    clip_data = filter_datatable_by_model_name(merged_table, "clip")
    vgg_data = filter_datatable_by_model_name(merged_table, "vgg")
    resnet_data = filter_datatable_by_model_name(merged_table, "resnet")

    models = {"clip": clip_data, "vgg16": vgg_data, "resnet50": resnet_data}
    colors = {"clip": 'm', "vgg16": 'y', "resnet50": 'c'}

    plt.figure()
    ax = plt.gca()
    x_ticks = None
    for i, model_name in enumerate(models):

        merged_table = add_mean_graph(["ballpark"], models[model_name], model_name)
        merged_table = add_std_graph(["ballpark"], merged_table)
        merged_table = add_max_graph(["ballpark"], merged_table)
        merged_table = add_min_graph(["ballpark"], merged_table)

        print(merged_table)

    # merged_table = merged_table.groupby(list(iter2path.keys()))['model_name'].apply(', '.join).reset_index()
    #
        merged_table = merged_table.set_index('model_name')
        merged_table = merged_table.transpose()



        for col in merged_table:
            if "ballpark" in col:
                print(col)

                ballpark_and_mean = merged_table.loc[:, merged_table.columns.intersection([col, "{}_svm_mean".format(model_name)])]


        # if ax is None:
        #     x_ticks = ballpark_and_mean.index
        #     ax = ballpark_and_mean.plot(y=list(ballpark_and_mean.columns), style=['-','--'], color = [colors[model_name], colors[model_name]])
        #     print("in")
        # else:
                x_ticks = ballpark_and_mean.index

                # ballpark_and_mean.plot(y=list(ballpark_and_mean.columns), style=['-', '--'],
                #                         color=[colors[model_name], colors[model_name]], ax=ax)

                for col in ballpark_and_mean.columns:
                    style = '--' if "svm" in col else "-"

                    if "svm" in col:
                        eb = ax.errorbar(x=x_ticks, y=ballpark_and_mean[col], yerr=merged_table["error"], label=col, c=colors[model_name], linestyle='--', capsize=2)
                    else:
                        eb = ax.errorbar(x=x_ticks, y=ballpark_and_mean[col], label=col, c=colors[model_name], linestyle="-")


                # ax.fill_between(x=range(len(x_ticks)), y1=merged_table["svm_min"], y2=merged_table["svm_max"])

                # ax.vlines(range(len(x_ticks)), merged_table["svm_min"], merged_table["svm_max"], color='k')

        # handles, labels = plt.gca().get_legend_handles_labels()
        # print(labels)
        # selected_handles = [handles[labels.index("ballpark_regress")]]
        # legend = plt.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=8)
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks)
    plt.legend(loc="lower right")

    # plt.ylim([0.25, 0.57])

    plt.ylabel("AUC")
    plt.xlabel(X_LABEL)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    plt.savefig(os.path.join(OUTPUT_PATH, "ballpark_and_mean"), bbox_inches='tight')
    plt.show()



    #
    # print(merged_table.columns)


    # merged_table = merged_table.set_index("index")



    # fig = plt.figure()
    # ax = merged_table.plot(y=list(merged_table.columns))
    # ax.set_xticks(range(len(merged_table.index)))
    # ax.set_xticklabels(merged_table.index)
    # # handles, labels = plt.gca().get_legend_handles_labels()
    # # print(labels)
    # # selected_handles = [handles[labels.index("ballpark_regress")]]
    # # legend = plt.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=8)
    # plt.ylabel("AUC")
    # plt.xlabel(X_LABEL)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #
    #
    # plt.savefig(os.path.join(OUTPUT_PATH, "all"), bbox_inches='tight')
    # plt.show()
    #
    # print(merged_table)
    #
    # ballpark_and_mean = merged_table.loc[:, merged_table.columns.intersection(["ballpark_regress", "svm_mean"])]
    #






if __name__=="__main__":
    main()

