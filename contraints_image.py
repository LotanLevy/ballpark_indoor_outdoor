


import argparse
import os
from affordance_tools.ContraintsParser import ConstraintsParser
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size



def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--constraints_file', '-cf',  type=str, required=True)

    parser.add_argument('--output_path', type=str, default=os.getcwd())
    return parser



def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print("Reads constraints file")
    constraints = ConstraintsParser(args.constraints_file)
    max_constrain_val = 1
    min_constraint_val = 0
    sorted_classes = dict()
    for cls in constraints.all_classes:
        sorted_classes[cls] = (min_constraint_val, max_constrain_val)
    for cls in constraints.lower_bounds:
        sorted_classes[cls] = (constraints.lower_bounds[cls],  sorted_classes[cls][1])
    for cls in constraints.upper_bounds:
        sorted_classes[cls] = (sorted_classes[cls][0], constraints.upper_bounds[cls])

    sorted_keys = [k for k, v in sorted(sorted_classes.items(), key=lambda item: item[0])] # sort by lower
    # sorted_keys = sorted(sorted_keys, key=lambda k: sorted_classes[k][1]) # sort by upper
    colors_values = [(sorted_classes[cls][0] + sorted_classes[cls][1])/2 for cls in sorted_keys]
    print(colors_values)
    print(sorted_keys)


    zipped_keys = []
    zipped_values = []
    cur = colors_values[0]
    cur_name = sorted_keys[0]
    for i in range(1, len(sorted_keys)):
        if cur is None:
            cur = colors_values[i]
            cur_name = sorted_keys[i]
            continue
        if colors_values[i] == cur:
            cur_name +="," + sorted_keys[i]
        else:
            zipped_keys.append([cur_name])
            zipped_values.append(cur)
            cur = colors_values[i]
            cur_name = sorted_keys[i]
    if cur is not None:
        zipped_keys.append([cur_name])
        zipped_values.append(cur)


    visualize_keys = sorted_keys
    visualize_values = colors_values

    visualize_keys = [[key] for key in visualize_keys]



    visualize_values = np.array(visualize_values).reshape(len(visualize_values), 1)



    print(visualize_keys)
    print(visualize_values)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(2, 5),
                           subplot_kw=dict(xticks=[], yticks=[]))

    cmap = plt.cm.get_cmap('Spectral')
    colorbar_colors = cmap(np.arange(cmap.N))
    colors = []
    for i in visualize_values:
        colors.append([cmap(i[0])])


    tb = plt.table(cellText=visualize_keys,
                   colLabels=['classes'],
                   loc='center',
                   cellLoc='center',
                   cellColours=colors)

    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    ax[1].add_table(tb)

    ax[0].imshow(colorbar_colors.reshape((1, colorbar_colors.shape[0], colorbar_colors.shape[1])), extent=[0, 1, 0, 0.1])
    ax[0].set_xticks(np.arange(0, 1.5, 0.5))
    a = fig.gca()
    a.set_frame_on(False)
    fig.tight_layout()
    # img.set_visible(False)
    plt.savefig(os.path.join(args.output_path, "constraints_image.jpg"), bbox_inches='tight')
    plt.show()


    print(sorted_classes)




if __name__ == "__main__":
    main()