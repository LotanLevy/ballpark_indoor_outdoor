
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Bag:
    def __init__(self, cls_name, data_iterator, cls_indices, bag_label, model, path2label_dict=None):
        self.cls_name = cls_name
        self.data_iterator = data_iterator
        self.cls_indices = cls_indices
        self.features_model = model
        self.bag_label = bag_label
        self.path2label_dict = path2label_dict



    def __len__(self):
        return len(self.cls_indices)

    def display_imagenet_prediction(self, idx):
        plt.close("all")
        plt.clf()
        preprocessed_image = self.data_iterator[idx][0]
        fig = plt.figure()
        pred = self.model.predict(preprocessed_image)
        plt.title("bag {} \n pred {} pred_label {}".format(self.cls_name, np.max(pred), np.argmax(pred)))
        orig = Image.open(self.data_iterator.filepaths[idx])
        plt.imshow(orig)
        plt.tight_layout()
        plt.show()
        plt.savefig("preds_for_cls_{}")
        plt.close(fig)

    def get_features(self, max_size=None):
        features_mat = None
        paths = []

        # self.display_imagenet_prediction(self.cls_indices[0])
        relevant_indices = self.cls_indices
        if max_size is not None:
            min_size = min([max_size, len(self.cls_indices)])
            relevant_indices = np.random.choice(self.cls_indices, min_size, replace=False)

        for i in relevant_indices:
            try:
                preprocessed_image = self.data_iterator[i][0]
            except:
                print("Problem in loading " + self.data_iterator.filepaths[i])
                continue
            paths.append(self.data_iterator.filepaths[i])


            features_vec = self.features_model(preprocessed_image)
            features_mat = features_vec if features_mat is None else np.concatenate((features_mat, features_vec))
        return features_mat, paths

    def get_true_labels(self):
        if self.path2label_dict is not None:
            labels = []
            for i in self.cls_indices:
                path = self.data_iterator.filepaths[i]
                labels.append(self.path2label_dict[path])
            return labels
        return None


    def display_bag(self):
        fig = plt.figure(figsize=(8, 8))
        plt.title(self.cls_name)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        columns = 4
        rows = 5
        for i, im_idx in zip(range(1, columns * rows + 1), self.cls_indices[:columns * rows + 1]):
            image = self.data_iterator[im_idx][0][0].astype(np.int)
            path = self.data_iterator.filepaths[im_idx]
            label = self.path2label_dict[path]
            fig.add_subplot(rows, columns, i)
            plt.title(label)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.imshow(image)
        plt.show()
        plt.close(fig)
        plt.clf()
        plt.cla()


