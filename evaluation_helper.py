
fn = 14
tp = 76
fp = 51
tn = 418

sum_data = fp + fn + tp + tn

accuracy = (tp + tn)/ sum_data

precision = tp / (tp+fp)
recall = tp / (tp + fn)
f_score = 2 * (precision *recall)/ (precision + recall)


print("accuracy: {}\nprecision: {}\nrecall: {}\nf-score: {}".format(accuracy,precision,recall,f_score))

