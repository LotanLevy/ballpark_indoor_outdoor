import pandas as pd
import matplotlib.pyplot as plt
import os



PATHS = "0$C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\bicycling\\bicycling_explore_2\\explore_0\\regress_results\\evaluations.csv," \
        "10$C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\bicycling\\bicycling_explore_2\\explore_1\\regress_results\\evaluations.csv," \
        "20$C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\bicycling\\bicycling_explore_2\\explore_2\\regress_results\\evaluations.csv," \
        "30$C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\bicycling\\bicycling_explore_2\\explore_3\\regress_results\\evaluations.csv"
        # "40$C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\bicycling\\bicycling_explore_2\\noise_40\\regress_results\\evaluations.csv"

GRAPH_FIELD = "auc"
X_LABEL = "Explore Iteration [Constraints Increasing]"

OUTPUT_PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\bicycling\\bicycling_explore_2"


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

def add_mean_graph(exclude_key_words, data_table):
    filtered_data_table = data_table
    for key in exclude_key_words:
        filtered_data_table = filtered_data_table[~filtered_data_table['model_name'].str.contains(key, na=True)]
    filtered_data_table = filtered_data_table.drop('model_name', axis=1)
    filtered_data_table = filtered_data_table.mean(axis=0).to_frame().transpose()
    filtered_data_table['model_name'] = pd.Series(["svm_mean"], index=filtered_data_table.index)
    # data_table.merge(filtered_data_table, how='left')
    filtered_data_table = data_table.merge(filtered_data_table, how="outer")
    return filtered_data_table



def main():
    paths = PATHS.split(",")
    iter2path = dict()
    for path in paths:
        param = path.split("$")
        iter2path[param[0]] = param[1]
    merged_table = merge_data(iter2path, GRAPH_FIELD)
    merged_table = add_mean_graph(["ballpark"], merged_table)
    merged_table = merged_table.groupby(list(iter2path.keys()))['model_name'].apply(', '.join).reset_index()

    merged_table = merged_table.set_index('model_name')
    merged_table = merged_table.transpose()

    print(merged_table.columns)


    # merged_table = merged_table.set_index("index")



    fig = plt.figure()
    ax = merged_table.plot(y=list(merged_table.columns))
    ax.set_xticks(range(len(merged_table.index)))
    ax.set_xticklabels(merged_table.index)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # print(labels)
    # selected_handles = [handles[labels.index("ballpark_regress")]]
    # legend = plt.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=8)
    plt.ylabel("AUC")
    plt.xlabel(X_LABEL)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


    plt.savefig(os.path.join(OUTPUT_PATH, "all"), bbox_inches='tight')
    plt.show()

    print(merged_table)

    ballpark_and_mean = merged_table.loc[:, merged_table.columns.intersection(["ballpark_regress", "svm_mean"])]



    plt.figure()
    ax = ballpark_and_mean.plot(y=list(ballpark_and_mean.columns))
    ax.set_xticks(range(len(ballpark_and_mean.index)))
    ax.set_xticklabels(ballpark_and_mean.index)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # print(labels)
    # selected_handles = [handles[labels.index("ballpark_regress")]]
    # legend = plt.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.ylabel("AUC")
    plt.xlabel(X_LABEL)
    plt.savefig(os.path.join(OUTPUT_PATH, "ballpark_and_mean"), bbox_inches='tight')
    plt.show()



if __name__=="__main__":
    main()

