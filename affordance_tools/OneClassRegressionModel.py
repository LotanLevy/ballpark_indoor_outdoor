import cvxpy as cp
import numpy as np
import itertools
from cvxpy.expressions.constants import Constant

NORM = 2
LOWER_BOUND_FOR_ONE_CLASS = 0.5
class OneClassRegressionModel:
    def __init__(self, constraints_parser, bags_dict, labeled_bags):
        self.constraints_parser = constraints_parser
        self.bags_dict = bags_dict
        self.labeled_bags = labeled_bags



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


    def add_bias(self, features_vector):
        bias_row = np.ones((features_vector.shape[0], 1))
        features_with_bias = np.concatenate((bias_row, features_vector), axis=1)
        return features_with_bias



    def solve_w_y(self, reg_val=10**-1, v=False, weights_path=None, reg_type="l2", output_path=None):
        if not self.legal_constraints:
            return None,None,None
        w = cp.Variable(self.features_num + 1)  # +intercept
        # reg = cp.square(cp.norm(w, NORM))
        yhat = cp.Variable(self.data_size)  # +intercept

        constraints = []
        loss = Constant(0)

        constraints.append(yhat >= 0)
        constraints.append(yhat <= 1)

        reg = cp.square(cp.norm(w, NORM))

        one_class_bag_indices = np.array([]).astype(np.int)
        one_class_bag_lower_bound = 0
        one_class_bag_upper_bound = 0


        for cls, bag_indices in self.bag2indices_range.items():
            bag = self.bags_dict[cls]
            if len(bag) == 0:
                print(cls + " is empty")
            if cls not in self.constraints_parser.all_classes:
                continue
            bag_features = self.add_bias(bag.get_features()[0])
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

            if lower_bound >= LOWER_BOUND_FOR_ONE_CLASS:
                one_class_bag_indices = np.concatenate([one_class_bag_indices, bag_indices])
                one_class_bag_lower_bound += lower_bound * np.shape(bag_indices)[0]
                one_class_bag_upper_bound += upper_bound * np.shape(bag_indices)[0]

        if len(one_class_bag_indices) > 0:
            one_class_bag_lower_bound /= len(one_class_bag_indices)
            one_class_bag_upper_bound /= len(one_class_bag_indices)
            print("one class bounds ({},{})".format(one_class_bag_lower_bound, one_class_bag_upper_bound))
            print(one_class_bag_indices)

            constraints.append((1. / (len(one_class_bag_indices))) * cp.sum(yhat[one_class_bag_indices]) >= one_class_bag_lower_bound)
            constraints.append((1. / (len(one_class_bag_indices))) * cp.sum(yhat[one_class_bag_indices]) <= one_class_bag_upper_bound)


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

        objective = loss/self.data_size + reg_val*reg


        # if self.labeled_bags is not None:
        #     positive_features = self.add_bias(self.labeled_bags["1"].get_features()[0])
        #     negative_features = self.add_bias(self.labeled_bags["0"].get_features()[0])
        #     print("constraints on {} labeled data".format(positive_features.shape[0] + negative_features.shape[0]))
        #
        #     labeled_loss = Constant(0)
        #
        #     pos_y = np.ones(positive_features.shape[0])
        #     neg_y = np.zeros(negative_features.shape[0])
        #
        #     labeled_loss += cp.sum_squares((positive_features * w) - pos_y)
        #     labeled_loss += cp.sum_squares((negative_features * w) - neg_y)
        #
        #     objective += 0.1 * (labeled_loss / (positive_features.shape[0] + negative_features.shape[0]))

        print(constraints)


        prob = cp.Problem(cp.Minimize(objective), constraints=constraints)

        try:
            prob.solve(verbose=v)
        except:
            prob.solve(solver="SCS")
        w_t = np.squeeze(np.asarray(np.copy(w.value)))
        y_t = np.squeeze(np.asarray(np.copy(yhat.value)))
        return w_t, y_t, prob.value

    def solve_w0(self):
        w = cp.Variable(self.features_num)  # +intercept






