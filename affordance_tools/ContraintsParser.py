import re


class ConstraintsParser:
    NUMBER_REGEX = r"[\s]*(([0-9]*[.])?[0-9]+)[\s]*"
    OBJECT_REGEX = r"([^<?|^>?]*)"
    DIFF_REGEX = r"^(.*?)-(.*?)$"
    # regexes
    RANGE_REGEX = "^" + NUMBER_REGEX + "<" + OBJECT_REGEX + "<" + NUMBER_REGEX + "$"
    INVERSE_RANGE_REGEX = "^" + NUMBER_REGEX + ">" + OBJECT_REGEX + ">" + NUMBER_REGEX + "$"
    LOWER_REGEX = "^" + NUMBER_REGEX + "<" + OBJECT_REGEX + "$"
    UPPER_REGEX = "^" + NUMBER_REGEX + ">" + OBJECT_REGEX + "$"
    INVERSE_LOWER_REGEX = "^" + OBJECT_REGEX + ">" + NUMBER_REGEX + "$"
    INVERSE_UPPER_REGEX = "^" + OBJECT_REGEX + "<" + NUMBER_REGEX + "$"
    def __init__(self, constraints_file):
        self.cls2cls_diff_lower_bounds = []# array of pair(pair, float)
        self.cls2cls_diff_upper_bounds = []# array of pair(pair, float)

        self.all_classes = set()

        self.lower_bounds = dict()
        self.upper_bounds = dict()

        self.constrained = set()

        with open(constraints_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.parse_line(line.strip())
        print(self.all_classes)

    def get_constraints_string(self):
        constraints_str = ""
        constraints_str += "diff_lowers " + str(self.cls2cls_diff_lower_bounds) + "\n"
        constraints_str += "diff_uppers " + str(self.cls2cls_diff_upper_bounds) + "\n"
        constraints_str += "lowers " + str(self.lower_bounds) + "\n"
        constraints_str += "uppers " + str(self.upper_bounds) + "\n"
        return constraints_str



    def parse_line(self, line):
        try:
            lower, cls_name, upper = self.parse_range(line)
            if lower is not None:
                lower = float(lower)
            if upper is not None:
                upper = float(upper)

            # m_diff = re.match(self.DIFF_REGEX, cls_name)
            classes = cls_name.split("-")
            print(cls_name)
            print(classes)
            # if m_diff is not None:
            if len(classes) > 1:
                # cls1, cls2 = m_diff.group(1).strip(), m_diff.group(2).strip()
                cls1, cls2 = classes[0].strip(), classes[1].strip()
                self.all_classes.add(cls1)
                self.all_classes.add(cls2)


                if lower is not None:
                    self.cls2cls_diff_lower_bounds.append(((cls1, cls2), lower))
                if upper is not None:
                    self.cls2cls_diff_upper_bounds.append(((cls1, cls2), upper))
                self.constrained.add(cls1)
                self.constrained.add(cls2)
            else:
                cls_name = cls_name.strip()
                self.all_classes.add(cls_name)

                if lower is not None:
                    self.lower_bounds[cls_name] = lower
                if upper is not None:
                    self.upper_bounds[cls_name] = upper
                self.constrained.add(cls_name)
        except Exception as e:
            print("problem in line " + line)
            raise e


    def parse_range(self, line):
        # matches
        m_range = re.match(self.RANGE_REGEX, line)
        m_inverse_range = re.match(self.INVERSE_RANGE_REGEX, line)
        m_lower = re.match(self.LOWER_REGEX, line)
        m_upper = re.match(self.UPPER_REGEX, line)
        m_inverse_lower = re.match(self.INVERSE_LOWER_REGEX, line)
        m_inverse_upper = re.match(self.INVERSE_UPPER_REGEX, line)

        # parsing
        if m_range is not None:
            lower, cls_name, upper = m_range.group(1), m_range.group(3), m_range.group(4)
        elif m_inverse_range is not None:
            lower, cls_name, upper = m_inverse_range.group(4), m_inverse_range.group(3), m_inverse_range.group(1)
        elif m_lower is not None:
            lower, cls_name, upper = m_lower.group(1), m_lower.group(3), None
        elif m_upper is not None:
            lower, cls_name, upper = None, m_upper.group(3), m_upper.group(1)
        elif m_inverse_lower is not None:
            lower, cls_name, upper = m_inverse_lower.group(2),  m_inverse_lower.group(1), None
        elif m_inverse_upper is not None:
            lower, cls_name, upper = None, m_inverse_upper.group(1), m_inverse_upper.group(2)
        else:
            return None, None, None
        return lower, cls_name, upper

    def get_negative_and_positive_classes_by_bound(self, bound): # lower bound first
        negative_classes, positive_classes = [], []
        print(bound[0], bound[1])
        for cls in self.upper_bounds:
            if self.upper_bounds[cls] <= bound[1]:
                negative_classes.append(cls)

        for cls in self.lower_bounds:
            if 1 - self.lower_bounds[cls] <= bound[0]:
                positive_classes.append(cls)
        return negative_classes, positive_classes







# c = ConstraintsParser("/indoor_outdoor_loose/Contraints_Examples.txt")
