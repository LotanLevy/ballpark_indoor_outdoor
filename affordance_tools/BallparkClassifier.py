

import cvxpy as cp
import numpy as np
import itertools
from cvxpy.expressions.constants import Constant
import os


class BallparkClassifier:
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
        bag_features, paths = bag.get_features()
        if bag_features is None:
            return None
        scores = (1. / len(bag)) * bag_features * theta
        return scores


    def solve_y(self, w, v=False):

        yhat = cp.Variable(self.data_size)  # +intercept

        constraints = []
        loss = Constant(0)
        constraints.append(yhat >= -1)
        constraints.append(yhat <= 1)

        for cls, bag_indices in self.bag2indices_range.items():
            bag = self.bags_dict[cls]
            if len(bag) == 0:
                print(cls + " is empty")
            bag_features, paths = bag.get_features()
            loss += cp.sum(cp.pos(1 - cp.multiply(yhat[bag_indices], bag_features @ w)))


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

    def solve_w(self, yhat, reg_val=10 ** -1, v=False):
        w = cp.Variable(self.features_num)  # +intercept
        reg = cp.square(cp.norm(w, 2))

        loss = Constant(0)

        for cls, bag_indices in self.bag2indices_range.items():
            bag = self.bags_dict[cls]
            if len(bag) == 0:
                print(cls + " is empty")
            bag_features, paths = bag.get_features()
            loss += cp.sum(cp.pos(1 - cp.multiply(yhat[bag_indices], bag_features @ w)))

        prob = cp.Problem(cp.Minimize(loss/self.data_size + reg_val*reg))

        try:
            prob.solve(verbose=v)
        except:
            prob.solve(solver="SCS")
        w_t = np.squeeze(np.asarray(np.copy(w.value)))
        return w_t, prob.value

    def get_w0(self, reg_val=10 ** -1, v=False):
        w = cp.Variable(self.features_num)  # +intercept
        reg = cp.square(cp.norm(w, 2))
        P = []
        for pair, lower_bound in self.constraints_parser.cls2cls_diff_lower_bounds:
            if lower_bound >= 0:
                P.append(pair)

        psi = cp.Variable(len(P))

        print(P)

        constraints = []
        if len(P) > 0:
            for idx, pair in enumerate(P):
                bag1, bag2 = self.bags_dict[pair[0]], self.bags_dict[pair[1]]
                bag1_features, paths = bag1.get_features()
                bag2_features, paths = bag2.get_features()

                features_sum1 = np.mean(bag1_features, axis=0)
                assert(len(features_sum1) == 4096)

                features_sum2 = np.mean(bag2_features, axis=0)
                assert(len(features_sum1) == 4096)

                constraints.append((features_sum1 @ w) >= (features_sum2 @ w) - psi[idx])


        prob = cp.Problem(cp.Minimize((cp.sum(psi) / len(P)) + reg_val * reg), constraints=constraints)

        try:
            prob.solve(verbose=v)
        except:
            prob.solve(solver="SCS")
        w_0 = np.squeeze(np.asarray(np.copy(w.value)))
        print(w_0.shape)
        return w_0, prob.value


    def solve_w_y(self, reg_val=10 ** -1, v=False, weights_path=None):
        if weights_path is not None and os.path.exists(weights_path + ".npy"):
            wt_1 = np.load(weights_path+ ".npy")
        else:
            wt_1, _ = self.get_w0(reg_val, v)
        t = 0
        while(True):
            print(wt_1)
            t += 1

            yt, _ = self.solve_y(wt_1, v=v)
            wt, _ = self.solve_w(np.sign(yt), reg_val=reg_val, v=v)
            diff = np.dot(wt-wt_1, wt-wt_1) / (np.dot(wt_1, wt_1) + 0.000001)
            print(diff)
            if (np.dot(wt-wt_1, wt-wt_1) / (np.dot(wt_1, wt_1) + 0.000001)) <= 10 **-5:
                print("exit")
                return wt, yt, None
            else:
                print("{}: end of solve w_y iteration with distance {}".format(t, np.dot(wt - wt_1, wt - wt_1) / (
                            np.dot(wt_1, wt_1) + 0.000001)))

                wt_1 = wt

                np.save(weights_path + "_{}".format(t), wt)






