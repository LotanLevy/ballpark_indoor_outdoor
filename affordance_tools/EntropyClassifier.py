

import cvxpy as cp
import numpy as np
import itertools
from cvxpy.expressions.constants import Constant
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
import os
from sklearn import svm


import tensorflow as tf
import datetime



class EntropyClassifier:
    def __init__(self, constraints_parser, bags_dict):
        self.constraints_parser = constraints_parser
        self.bags_dict = bags_dict

        self.pairwise_bags = list(itertools.combinations(list(self.bags_dict.keys()),2))


        assert len(self.bags_dict) > 0

        self.features_num = list(self.bags_dict.values())[0].features_model.layers[-1].output.shape[1]
        self.data_size, self.bag2indices_range = self.get_data_size()

        self.legal_constraints = self._check_constraints_with_bags()

        print("Initialize ballpark parameters")
        print("Bags names {}".format(self.bags_dict.keys()))
        print("Constraints {}".format(self.constraints_parser.get_constraints_string()))
        print("constraints are legal - {}".format(self.legal_constraints))
        print("w size {}".format(self.features_num))
        print("y size {}".format(self.data_size))


    def get_data_size(self):
        length = 0
        bag2indices_range = dict()
        for bag_name, bag in self.bags_dict.items():
            bag2indices_range[bag_name] = range(length, length+len(bag))
            length += len(bag)
        return length, bag2indices_range

    def _check_constraints_with_bags(self):
        legal_constraints = True
        for cls in self.constraints_parser.constrained:
            if cls not in self.bags_dict:
                print("Illegal constraint's class {}".format(cls))
                legal_constraints = False
                break
        return legal_constraints


    def _get_score_by_cls_name(self, cls_name, theta):
        bag = self.bags_dict[cls_name]
        bag_features, paths = self.get_bag_features_with_bias(bag)
        if bag_features is None:
            return None
        scores = (1. / len(bag)) * bag_features * theta
        return scores



    # def softmax(self, x):
    #     # first shift the values of f so that the highest number is 0:
    #     x -= np.max(x)  # f becomes [-666, -333, 0]
    #     p = np.exp(x) / np.sum(np.exp(x), axis=1)  # safe to do, gives the correct answer
    #     return p


    def solve_y(self, W, v=False):

        yhat = cp.Variable(self.data_size)  # +intercept

        constraints = []
        loss = Constant(0)
        constraints.append(yhat >= 0)
        constraints.append(yhat <= 1)

        for cls, bag_indices in self.bag2indices_range.items():
            bag = self.bags_dict[cls]
            if len(bag) == 0:
                print(cls + " is empty")
            bag_features, paths = self.get_bag_features_with_bias(bag)
            preds = np.dot(W, bag_features.T)



            loss += cp.sum(-cp.multiply(yhat[bag_indices], cp.log(preds))-cp.multiply(1-yhat[bag_indices], cp.log(1-preds)))

            # upper and lower constraints
            if cls in self.constraints_parser.lower_bounds:
                lower_bound = self.constraints_parser.lower_bounds[cls]
            else:
                lower_bound = 0
            constraints.append((1. / (len(bag))) * cp.sum(yhat[bag_indices]) >= lower_bound)

            if cls in self.constraints_parser.upper_bounds:
                upper_bound = self.constraints_parser.upper_bounds[cls]
            else:
                upper_bound = 1
            constraints.append((1. / (len(bag))) * cp.sum(yhat[bag_indices]) <= upper_bound)

        # pairwise constraints
        for pair, lower_bound in self.constraints_parser.cls2cls_diff_lower_bounds:
            high_bag, low_bag = self.bags_dict[pair[0]], self.bags_dict[pair[1]]
            high_bag_idx_range, low_bag_idx_range = self.bag2indices_range[pair[0]], self.bag2indices_range[pair[1]]

            constraints.append((1. / (len(high_bag))) * cp.sum(yhat[high_bag_idx_range]) -
                               (1. / (len(low_bag))) * cp.sum(yhat[low_bag_idx_range]) >= lower_bound)

        for pair, upper_bound in self.constraints_parser.cls2cls_diff_upper_bounds:
            high_bag, low_bag = self.bags_dict[pair[0]], self.bags_dict[pair[1]]
            high_bag_idx_range, low_bag_idx_range = self.bag2indices_range[pair[0]], self.bag2indices_range[pair[1]]

            constraints.append((1. / (len(high_bag))) * cp.sum(yhat[high_bag_idx_range]) -
                               (1. / (len(low_bag))) * cp.sum(yhat[low_bag_idx_range]) <= upper_bound)

        prob = cp.Problem(cp.Minimize(loss / self.data_size), constraints=constraints)
        try:
            prob.solve(verbose=v)
        except:
            prob.solve(solver="SCS")
        y_t = np.squeeze(np.asarray(np.copy(yhat.value)))
        return y_t, prob.value

    def get_bag_features_with_bias(self, bag):
        bag_features, paths = bag.get_features()
        bias_row = np.ones((bag_features.shape[0], 1))
        bag_features = np.concatenate((bias_row, bag_features), axis=1)
        return bag_features, paths



    def solve_w(self, yhat, reg_val=10 ** -1, v=False):
        w = cp.Variable(self.features_num+1)  # +intercept
        reg = cp.square(cp.norm(w, 2))

        loss = Constant(0)
        constraints = []


        for cls, bag_indices in self.bag2indices_range.items():
            bag = self.bags_dict[cls]
            if len(bag) == 0:
                print(cls + " is empty")
            bag_features, paths = self.get_bag_features_with_bias(bag)
            preds = bag_features @ w
            loss += cp.sum(
                -cp.multiply(yhat[bag_indices], cp.log(preds)) - cp.multiply(1 - yhat[bag_indices], cp.log(1 - preds)))

            constraints.append((bag_features @ w) >= 0)
            constraints.append((bag_features @ w) <= 1)

        # for cls, bag_indices in self.bag2indices_range.items():
        #     bag = self.bags_dict[cls]
        #     if len(bag) == 0:
        #         print(cls + " is empty")
        #     bag_features, paths = self.get_bag_features_with_bias(bag)
        #     constraints.append((bag_features @ w) >= 0)
        #     constraints.append((bag_features @ w) <= 1)

        prob = cp.Problem(cp.Minimize(loss/self.data_size + reg_val*reg), constraints=constraints)

        try:
            prob.solve(verbose=v)
        except:
            prob.solve(solver="SCS")
        w_t = np.squeeze(np.asarray(np.copy(w.value)))
        return w_t



    def get_Pset(self):
        P = []
        for pair, lower_bound in self.constraints_parser.cls2cls_diff_lower_bounds:
            if lower_bound >= 0:
                P.append(pair)
        return P




    def get_w0(self, reg_val=10 ** -1, v=False):
        w = cp.Variable(self.features_num + 1)  # +intercept
        reg = cp.square(cp.norm(w, 2))
        P = self.get_Pset()
        psi = cp.Variable(len(P))
        constraints = []
        if len(P) > 0:
            for idx, pair in enumerate(P):
                bag1, bag2 = self.bags_dict[pair[0]], self.bags_dict[pair[1]]
                bag1_features, paths = self.get_bag_features_with_bias(bag1)
                bag2_features, paths = self.get_bag_features_with_bias(bag2)

                features_mean1 = np.sum(bag1_features, axis=0)/ len(bag1)
                assert((features_mean1 == np.mean(bag1_features, axis=0)).all())
                features_mean2 = np.sum(bag2_features, axis=0)/ len(bag2)
                assert((features_mean2 == np.mean(bag2_features, axis=0)).all())

                constraints.append((features_mean1 @ w) >= (features_mean2 @ w) - psi[idx])
        else:
            print("_________________________________")
            print("P set is empty, w0=0")
        for cls, bag_indices in self.bag2indices_range.items():
            bag = self.bags_dict[cls]
            if len(bag) == 0:
                print(cls + " is empty")
            bag_features, paths = self.get_bag_features_with_bias(bag)
            constraints.append((bag_features @ w) >= 0)
            constraints.append((bag_features @ w) <= 1)


        prob = cp.Problem(cp.Minimize((cp.sum(psi) / len(P)) + reg_val * reg), constraints=constraints)

        try:
            prob.solve(verbose=v)
        except:
            prob.solve(solver="SCS")
        w_0 = np.squeeze(np.asarray(np.copy(w.value)))
        return w_0, prob.value

    def test_on_train(self, w, constraints_limit = 0.3):
        train_features = None
        all_labels = np.array([])
        all_paths = []

        for cls, bag_indices in self.bag2indices_range.items():

            # upper and lower constraints
            if cls in self.constraints_parser.lower_bounds:
                bag_lower_bound = self.constraints_parser.lower_bounds[cls]
            else:
                bag_lower_bound = 0

            if cls in self.constraints_parser.upper_bounds:
                bag_upper_bound = self.constraints_parser.upper_bounds[cls]
            else:
                bag_upper_bound = 1
            bag = self.bags_dict[cls]
            if len(bag) == 0:
                print(cls + " is empty")
            if 1 - bag_lower_bound <= constraints_limit:
                bag_features, paths = self.get_bag_features_with_bias(bag)
                all_labels = np.concatenate((all_labels, np.ones(len(bag))))
                if train_features is None:
                    train_features = bag_features
                else:
                    train_features = np.concatenate((train_features, bag_features))
                all_paths += paths
            elif bag_upper_bound <= constraints_limit:
                bag_features, paths = self.get_bag_features_with_bias(bag)
                all_labels = np.concatenate((all_labels, -1 * np.ones(len(bag))))
                if train_features is None:
                    train_features = bag_features
                else:
                    train_features = np.concatenate((train_features, bag_features))
                all_paths += paths
            assert train_features.shape[0] == all_labels.shape[0]

        preds = np.around( np.dot(w, train_features.T))
        preds[np.where(preds==0)] = -1
        print(all_labels.shape)
        print(preds.shape)

        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        sum_data = fp + fn + tp + tn

        accuracy = (tp + tn) / sum_data

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * (precision * recall) / (precision + recall)
        print("accuracy: {}\nprecision: {}\nrecall: {}\nfscore: {}\n".format(accuracy, precision, recall, f_score))
        return accuracy, precision, recall, f_score









    def solve_w_y(self, reg_val=10 ** -1, v=False, weights_path=None, reg_type=None, output_path=os.getcwd()):
        # if weights_path is not None and os.path.exists(weights_path + ".npy"):
        #     wt_1 = np.load(weights_path+ ".npy")
        # else:
        wt_1, _ = self.get_w0(reg_val, v)
        t = 0
        metrics = ['accuracy', 'precision', 'recall', 'f_score']
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(output_path, 'logs/' + current_time )
        train_summary_writer = tf.summary.create_file_writer(log_dir)

        accuracy, precision, recall, f_score = self.test_on_train(wt_1, constraints_limit=0.2)
        with train_summary_writer.as_default():
            tf.summary.scalar('accuracy', accuracy, step=t)
            tf.summary.scalar('precision', precision, step=t)
            tf.summary.scalar('recall', recall, step=t)
            tf.summary.scalar('f_score', f_score, step=t)
            tf.summary.scalar('diff', np.inf, step=t)



        while(True):
            t += 1
            yt, _ = self.solve_y(wt_1, v=v)
            wt = self.solve_w(np.round(yt), reg_val=reg_val, v=v)
            diff = np.dot(wt-wt_1, wt-wt_1) / (np.dot(wt_1, wt_1) + 0.000001)
            # test the loss on training data
            accuracy, precision, recall, f_score = self.test_on_train(wt, constraints_limit=0.2)
            with train_summary_writer.as_default():
                tf.summary.scalar('accuracy', accuracy, step=t)
                tf.summary.scalar('precision', precision, step=t)
                tf.summary.scalar('recall', recall, step=t)
                tf.summary.scalar('f_score', f_score, step=t)
                tf.summary.scalar('diff', diff, step=t)


            if diff <= 10 **-5:
                print("exit")

                return wt, yt, None
            else:
                print("{}: end of solve w_y iteration with distance {}".format(t, np.dot(wt - wt_1, wt - wt_1) / (
                            np.dot(wt_1, wt_1) + 0.000001)))

                wt_1 = wt

                np.save(weights_path + "_{}".format(t), wt)






