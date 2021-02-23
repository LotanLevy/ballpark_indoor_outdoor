import cvxpy as cp
import numpy as np
import itertools
from cvxpy.expressions.constants import Constant

NORM = 2

class RegressionModel:
    def __init__(self, constraints_parser, bags_dict):
        self.constraints_parser = constraints_parser
        self.bags_dict = bags_dict



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

    def feasibility_regression(self):
        if not self.legal_constraints:
            return None
        theta = cp.Variable(self.features_num)
        reg = cp.square(cp.norm(theta, NORM))

        constraints = []
        for pair, lower_bound in self.constraints_parser.cls2cls_diff_lower_bounds:
            scores_high = self._get_score_by_cls_name(pair[0], theta)
            scores_low = self._get_score_by_cls_name(pair[1], theta)
            constraints.append(cp.sum(scores_high) - cp.sum(scores_low) > lower_bound)

        for pair, upper_bound in self.constraints_parser.cls2cls_diff_upper_bounds:
            scores_high = self._get_score_by_cls_name(pair[0], theta)
            scores_low = self._get_score_by_cls_name(pair[1], theta)
            constraints.append(cp.sum(scores_high) - cp.sum(scores_low) < upper_bound)

        for cls in self.bags_dict:
            scores = self._get_score_by_cls_name(cls, theta)
            if cls in self.constraints_parser.lower_bounds:
                lower_bound = self.constraints_parser.lower_bounds[cls]
                constraints.append(cp.sum(scores) <= lower_bound)
            if cls in self.constraints_parser.upper_bounds:
                upper_bound = self.constraints_parser.upper_bounds[cls]
                constraints.append(cp.sum(scores) <= upper_bound)

        prob = cp.Problem(cp.Minimize(1 * reg), constraints=constraints)

        try:
            prob.solve(verbose=False)
        except:
            prob.solve(solver="SCS")
        w_t = np.squeeze(np.asarray(np.copy(theta.value)))
        return w_t

    def solve_w_y(self, reg_val=10**-1, v=False, weights_path=None, reg_type="l2", output_path=None):
        if not self.legal_constraints:
            return None,None,None
        w = cp.Variable(self.features_num)  # +intercept
        # reg = cp.square(cp.norm(w, NORM))
        yhat = cp.Variable(self.data_size)  # +intercept

        constraints = []
        loss = Constant(0)

        constraints.append(yhat >= 0)
        constraints.append(yhat <= 1)

        if reg_type == "l2":
            reg = cp.square(cp.norm(w, NORM))
        elif reg_type == "entropy":
            reg = -cp.sum(cp.entr(w))
        else:
            print("wrong reg")
            return None, None, None
        print(reg_type, reg)


        for cls, bag_indices in self.bag2indices_range.items():
            bag = self.bags_dict[cls]
            if len(bag) == 0:
                print(cls + " is empty")
            if cls not in self.constraints_parser.all_classes:
                continue
            bag_features, paths = bag.get_features()

            # loss += cp.sum_squares((bag_features * w)-yhat[bag_indices])
            loss += cp.sum_squares((bag_features * w)-yhat[bag_indices])


            if cls in self.constraints_parser.lower_bounds:
                lower_bound = self.constraints_parser.lower_bounds[cls]
            else:
                lower_bound = 0
            constraints.append((1./(len(bag)))*cp.sum(yhat[bag_indices]) >= lower_bound)

            if cls in self.constraints_parser.upper_bounds:
                upper_bound = self.constraints_parser.upper_bounds[cls]
            else:
                upper_bound = 1
            constraints.append((1./(len(bag)))*cp.sum(yhat[bag_indices]) <= upper_bound)

        for pair, lower_bound in self.constraints_parser.cls2cls_diff_lower_bounds:
            high_bag, low_bag = self.bags_dict[pair[0]], self.bags_dict[pair[1]]
            high_bag_idx_range, low_bag_idx_range = self.bag2indices_range[pair[0]], self.bag2indices_range[pair[1]]

            constraints.append((1./(len(high_bag)))*cp.sum(yhat[high_bag_idx_range]) -
                               (1./(len(low_bag)))*cp.sum(yhat[low_bag_idx_range]) >= lower_bound)

        for pair, upper_bound in self.constraints_parser.cls2cls_diff_upper_bounds:
            high_bag, low_bag = self.bags_dict[pair[0]], self.bags_dict[pair[1]]
            high_bag_idx_range, low_bag_idx_range = self.bag2indices_range[pair[0]], self.bag2indices_range[pair[1]]

            constraints.append((1. / (len(high_bag))) * cp.sum(yhat[high_bag_idx_range]) -
                               (1. / (len(low_bag))) * cp.sum(yhat[low_bag_idx_range]) <= upper_bound)

        prob = cp.Problem(cp.Minimize(loss/self.data_size + reg_val*reg), constraints=constraints)

        try:
            prob.solve(verbose=v)
        except:
            prob.solve(solver="SCS")
        w_t = np.squeeze(np.asarray(np.copy(w.value)))
        y_t = np.squeeze(np.asarray(np.copy(yhat.value)))
        return w_t, y_t, prob.value

    def solve_w0(self):
        w = cp.Variable(self.features_num)  # +intercept






