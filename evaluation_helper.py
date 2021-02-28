import matplotlib.pyplot as plt
import os
import numpy as np
#
# OUTPUT_PATH = "C:\\Users\\lotan\\Documents\\studies\\phoenix\\ballpark_experiments\\paper_experiments\\dine_noise"
#
# EER = {"svm0":[0.2, 0.21],
#        "svm1":[0.188, 0.21],
#        "svm2":[0.2, 0.19],
#        "svm3":[0.165, 0.17],
#        "svm4":[0.19, 0.19]}
# ROC = {"svm0":[0.88, 0.88],
#        "svm1":[0.9, 0.88],
#        "svm2":[0.9, 0.88],
#        "svm3":[0.91, 0.91],
#        "svm4":[0.88, 0.88]}
# fn_fp_tn_tp = {"svm0":[(26,86,334,101), (27,89,331,100), (), (), ()],
#                "svm1":[(24,79,341,103), (27,89,331,100), (), (), ()],
#                "svm2":[(26,86,336,101), (25,81,339,102), (), (), ()],
#                "svm3":[(21,70,350,106), (22,73,347,105), (), (), ()],
#                "svm4":[(24,81,339,103), (25,85,335,102), (), (), ()]}
#
# f_scores = {}
# for model_name, tuples_list in fn_fp_tn_tp.items():
#     f_sores_list = []
#     for tuple_ in tuples_list:
#         fp, fn, tp, tn = tuple_[0], tuple_[1], \
#                          tuple_[2], tuple_[3]
#         sum_data = fp + fn + tp + tn
#
#         accuracy = (tp + tn)/ sum_data
#
#         precision = tp / (tp+fp)
#         recall = tp / (tp + fn)
#         f_score = 2 * (precision *recall)/ (precision + recall)
#         f_sores_list.append(f_score)
#     f_scores[model_name] = f_sores_list
#
# arrays_to_plot = {"EER":EER, "ROC": ROC, "F-score": f_scores}
# avg_to_plot = {"EER":[], "ROC": [], "F-score": []}
#
# for method in arrays_to_plot:
#     avg_values = [0, 0, 0, 0, 0]
#     for key in arrays_to_plot[method]:
#         for i, value in enumerate(arrays_to_plot[method][key]):
#             avg_values[i] += value
#     avg_to_plot[method] = np.array(avg_values)/5
#
# print(avg_to_plot)
# X_VALUES = list(range(0,50,10))
#
# for evaluation_name, evaluation_map in arrays_to_plot.items():
#     fig = plt.figure()
#     for model_name, values in evaluation_map.items():
#         plt.plot(X_VALUES, values, label=model_name)
#     plt.legend()
#     plt.title(evaluation_name)
#     plt.xlabel("noise")
#     plt.ylabel(evaluation_name + " values")
#     plt.savefig(os.path.join(OUTPUT_PATH, evaluation_name+"_noise_impact"))
#


fn = 12
fp = 39
tn = 430
tp = 82


sum_data = fp + fn + tp + tn

accuracy = (tp + tn)/ sum_data

precision = tp / (tp+fp)
recall = tp / (tp + fn)
f_score = 2 * (precision *recall)/ (precision + recall)


print("accuracy: {}\nprecision: {}\nrecall: {}\nf-score: {}".format(accuracy,precision,recall,f_score))

