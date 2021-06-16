
import pandas as pd
import matplotlib.pyplot as plt
import os
runs_evaluations_paths={"50":"C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\ssvm\\svm_with_25_labeled_examples\\regress_results\\evaluations.csv",
                        "100":"C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\ssvm\\svm_with_50_labeled_examples\\regress_results\\evaluations.csv",
                        "200":"C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\ssvm\\svm_with_100_labeled_examples\\regress_results\\evaluations.csv",
                        "300":"C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\ssvm\\svm_with_150_labeled_examples\\regress_results\\evaluations.csv",
                        "400":"C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\ssvm\\svm_with_200_labeled_examples\\regress_results\\evaluations.csv",
                        "600":"C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\ssvm\\svm_with_300_labeled_examples\\regress_results\\evaluations.csv",
                        "800":"C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\ssvm\\svm_with_400_labeled_examples\\regress_results\\evaluations.csv"}


field_to_parse = "auc"
output_path = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\experiment_outputs\\clip\\buy\\manual_constraints\\ssvm\\evaluations_files"

if not os.path.exists(output_path):
    os.makedirs(output_path)

evaluations_for_runs = dict()


for run_num, run in runs_evaluations_paths.items():
    df = pd.read_csv(run)    new_df = pd.DataFrame(df.model_name.str.split('_',1).tolist(),
                                 columns = ['model','model_name_rest'])
    df["model"] = new_df["model"]

    evaluation_dict = dict()

    unique_models = df["model"].unique()
    for model in unique_models:
        avg = df[df["model"] == model][field_to_parse].mean().round(2)
        std = df[df["model"] == model][field_to_parse].std().round(3)
        evaluation_dict[model] = (avg, std)
    evaluations_for_runs[run_num] = evaluation_dict

with open(os.path.join(output_path, "results_dict.txt"), 'w') as f:
    f.write(str(evaluations_for_runs))

models_data_labels_num = dict()
models_data_means = dict()
models_data_std = dict()

for labels_num in evaluations_for_runs:
    for model in evaluations_for_runs[labels_num]:
        if model not in models_data_labels_num:
            models_data_labels_num[model] = []
            models_data_means[model] = []
            models_data_std[model] = []
        avg, std = evaluations_for_runs[labels_num][model]
        models_data_labels_num[model].append(labels_num)
        models_data_means[model].append(avg)
        models_data_std[model].append(std)

for model in models_data_labels_num:
    plt.figure()
    plt.errorbar(models_data_labels_num[model], models_data_means[model], yerr=models_data_std[model])
    plt.title(model)
    plt.xlabel("labels number")
    plt.ylabel("auc")
    plt.savefig(os.path.join(output_path, model))
    plt.show()




print(evaluations_for_runs)
