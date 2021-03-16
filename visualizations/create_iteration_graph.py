import pandas as pd
import matplotlib.pyplot as plt



PATHS = "iter0$C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\buy\\buy_explore\\explore_0\\regress_results\\evaluations.csv," \
        "iter1$C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\buy\\buy_explore\\explore_1\\regress_results\\evaluations.csv," \
        "iter2$C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\buy\\buy_explore\\explore_2\\regress_results\\evaluations.csv," \
        "iter3$C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\buy\\buy_explore\\explore_3\\regress_results\\evaluations.csv"
        # "iter4$C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\new_experiments\\buy\\buy_explore\\explore_4\\regress_results\\evaluations.csv"

GRAPH_FIELD = "auc"


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

def main():
    paths = PATHS.split(",")
    iter2path = dict()
    for path in paths:
        param = path.split("$")
        iter2path[param[0]] = param[1]
    merged_table = merge_data(iter2path, GRAPH_FIELD)
    merged_table = merged_table.groupby(list(iter2path.keys()))['model_name'].apply(', '.join).reset_index()

    merged_table = merged_table.set_index('model_name')
    print(merged_table)
    merged_table = merged_table.transpose()

    # merged_table = merged_table.set_index("index")
    print(merged_table.index)



    plt.figure()
    ax = merged_table.plot(y=list(merged_table.columns))
    ax.set_xticks(range(len(merged_table.index)))
    ax.set_xticklabels(merged_table.index)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # print(labels)
    # selected_handles = [handles[labels.index("ballpark_regress")]]
    legend = plt.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=8)
    plt.show()



if __name__=="__main__":
    main()

