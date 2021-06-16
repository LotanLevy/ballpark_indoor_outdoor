
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



PATHS= {"2071 imgs": "C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\dine\\manual_constraints\\ballpark_polar_supervised\\explore_3\\regress_results\\evaluations.csv",
        "2009 imgs":"C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\dine\\manual_constraints\\ballpark_clean_polar_supervised\\explore_3\\regress_results\\evaluations.csv",
        "1976 imgs":"C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\dine\\manual_constraints\\ballpark_clean_polar_supervised_2\\explore_3\\regress_results\\evaluations.csv"}


GRAPH_FIELD = "auc"
X_LABEL = "ballpark-psvm auc"

OUTPUT_PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\dine\\manual_constraints\\ballpark_polar_supervised\\compare_supervision_3"

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


def parse_model_name_to_columns(table):
    model_type, bounds = [], []
    for model_name in table["model_name"]:
        values = model_name.split("_")
        model_type.append(values[1])
        min_b, max_b = values[3], values[4]
        if min_b == "no" and max_b == "labels":
            min_b, max_b = 0, 0
        bounds.append("{},{}".format(min_b, max_b ))
    table["model"] = model_type
    table["bounds"] = bounds
    return table

def merge_bounds(table):
    print(table)
    unique_values = table[["ballpark", "svm"]].drop_duplicates()
    bounds, ballpark_values, svm_values = [], [], []
    for i, row in unique_values.T.iteritems():
        print(row)
        all_values = table[(table["ballpark"] == row["ballpark"]) & (table["svm"] == row["svm"])]
        bounds.append(",".join(["({})".format(b) for b in all_values["bounds"]]))
        ballpark_values.append(row["ballpark"])
        svm_values.append(row["svm"])


    new_table = pd.DataFrame(data={"bounds":bounds, "svm":svm_values, "ballpark":ballpark_values})
    return new_table



def create_table_by_bounds(table):

    new_table = pd.DataFrame(data={"bounds":pd.unique(table["bounds"])})
    bounds, ballpark_values, svm_values = [], [], []
    for bound in new_table["bounds"]:
        filtered_rows = table[table["bounds"] == bound]
        ballpark_value = filtered_rows[filtered_rows["model"] == "ballpark"][GRAPH_FIELD]
        if "svm" not in list(filtered_rows["model"]):
            continue
        svm_value = filtered_rows[filtered_rows["model"] == "svm"][GRAPH_FIELD]
        bounds.append(bound)
        ballpark_values.append(ballpark_value.values[0])
        svm_values.append(svm_value.values[0])
    #
    # # merge equal bounds
    # values_dict = dict()
    # for bound, ballpark_val, svm_val in zip(bounds, ballpark_values, svm_values):
    #     if ballpark_val not in values_dict:
    #         values_dict[ballpark_val] = dict()
    #     if svm_val not in values_dict[ballpark_val]:
    #         values_dict[ballpark_val][svm_val] = bound
    #     else:
    #         values_dict[ballpark_val][svm_val] = values_dict[ballpark_val][svm_val] + "," + bound


    new_table = pd.DataFrame(data={"bounds":bounds, "svm":svm_values, "ballpark":ballpark_values})

    return new_table


def create_roc_graph(output_path, iter_params):
    # roc_auc = auc(values_x, values_y)

    min_val = 1
    max_val = 0

    plt.figure()
    lw = 0.1
    for iter_name in iter_params:
        bounds, values_x, values_y, no_label_ballpark, color = iter_params[iter_name]
        plt.scatter(values_x, values_y, color=color,
                 lw=lw, label='ballpark vs psvm [{}]'.format(iter_name))
        for txt, x, y in zip(bounds, values_x, values_y):
            plt.annotate(txt, (x, y), fontsize=6)
        min_val = min(min_val, min(min(values_x), min(values_y)))
        max_val = max(max_val, max(max(values_x), max(values_y)), no_label_ballpark)

    for iter_name in iter_params:
        bounds, values_x, values_y, no_label_ballpark, color = iter_params[iter_name]
        plt.plot([min_val, max_val], [no_label_ballpark, no_label_ballpark], color=color, lw=2, linestyle='-', label="ballpark with no labels [{}]".format(iter_name))
    plt.plot([min_val, max_val], [min_val, max_val], color="k", lw=2, linestyle='--', label="equal performance")

    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.ylabel('ballpark auc')
    plt.xlabel('polar svm auc')
    plt.legend(loc="lower right")
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()



def parse_data_for_model(data_table, model_name):
    data = filter_datatable_by_model_name(data_table, model_name)
    data = parse_model_name_to_columns(data)
    bounds_compare = create_table_by_bounds(data)
    bounds_compare = merge_bounds(bounds_compare)
    return bounds_compare, data


def create_iter_params_dict(iters_for_model, colors):
    iters_params = dict()
    for i, iter in enumerate(iters_for_model):
        model_params, model_data = iters_for_model[iter]
        iters_params[iter] = (model_params["bounds"].values,
                                model_params["svm"].values,
                                model_params["ballpark"].values,
                                model_data[model_data["bounds"] == "0,0"][GRAPH_FIELD].values[0],
                                colors[i])
    return iters_params


def main():

    iter_params_dict = dict()

    models_dict = dict()
    models_dict["clip"] = dict()
    models_dict["vgg"] = dict()
    models_dict["resnet"] = dict()

    for iter_name, iter_path in PATHS.items():
        table = pd.read_csv(iter_path)[["model_name", GRAPH_FIELD]]
        #
        # clip_data = filter_datatable_by_model_name(table, "clip")
        # vgg_data = filter_datatable_by_model_name(table, "vgg")
        # resnet_data = filter_datatable_by_model_name(table, "resnet")
        #
        # clip_data = parse_model_name_to_columns(clip_data)
        # vgg_data = parse_model_name_to_columns(vgg_data)
        # resnet_data = parse_model_name_to_columns(resnet_data)
        #
        # clip_bounds_compare = create_table_by_bounds(clip_data)
        # vgg_bounds_compare = create_table_by_bounds(vgg_data)
        # resnet_bounds_compare = create_table_by_bounds(resnet_data)

        models_dict["clip"][iter_name] = parse_data_for_model(table, "clip")
        models_dict["vgg"][iter_name] = parse_data_for_model(table, "vgg")
        models_dict["resnet"][iter_name] = parse_data_for_model(table, "resnet")


    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    get_model_iters_dict = lambda model_params, model_data : {"iter 1": (model_params["bounds"].values,
                            model_params["svm"].values,
                            model_params["ballpark"].values,
                            model_data[model_data["bounds"] == "0,0"][GRAPH_FIELD].values[0],
                            "c")}

    colors = ["c", "y", "m"]

    create_roc_graph(os.path.join(OUTPUT_PATH, "vgg16_supervision.png"), create_iter_params_dict(models_dict["vgg"], colors))

    create_roc_graph(os.path.join(OUTPUT_PATH, "resnet50_supervision.png"), create_iter_params_dict(models_dict["resnet"], colors))
    # resnet_bounds_compare["bounds"].values, resnet_bounds_compare["svm"].values, resnet_bounds_compare["ballpark"].values, resnet_data[resnet_data["bounds"] == "0,0"][GRAPH_FIELD].values[0])
    create_roc_graph(os.path.join(OUTPUT_PATH, "clip_supervision.png"), create_iter_params_dict(models_dict["clip"], colors))
    # clip_bounds_compare["bounds"].values, clip_bounds_compare["svm"].values, clip_bounds_compare["ballpark"].values, clip_data[clip_data["bounds"] == "0,0"][GRAPH_FIELD].values[0])












    #
    # models = {"clip": clip_data, "vgg16": vgg_data, "resnet50": resnet_data}
    # colors = {"clip": 'm', "vgg16": 'y', "resnet50": 'c'}
    #
    # plt.figure()
    # ax = plt.gca()
    # x_ticks = None
    # for i, model_name in enumerate(models):
    #
    #     merged_table = add_mean_graph(["ballpark"], models[model_name], model_name)
    #     merged_table = add_std_graph(["ballpark"], merged_table)
    #     merged_table = add_max_graph(["ballpark"], merged_table)
    #     merged_table = add_min_graph(["ballpark"], merged_table)
    #
    #     print(merged_table)
    #
    # # merged_table = merged_table.groupby(list(iter2path.keys()))['model_name'].apply(', '.join).reset_index()
    # #
    #     merged_table = merged_table.set_index('model_name')
    #     merged_table = merged_table.transpose()


main()