
import argparse
import pandas as pd
import os
from shutil import copyfile


def get_args_parser():
    parser = argparse.ArgumentParser(description='Process constraint collector args.')
    parser.add_argument('--paths2scores1', '-s1', type=str, required=True)
    parser.add_argument('--paths2scores2', '-s2', type=str, required=True)
    parser.add_argument('--name1', '-n1', type=str, required=True)
    parser.add_argument('--name2', '-n2', type=str, required=True)

    parser.add_argument('--output_path',  '-o', type=str)
    return parser



def copy_images_of_df(df, output_path):
    df = df.sort_values(by=[1])
    print(df[1].tolist())

    print(df[0].tolist())
    for i, path in enumerate(df[0].tolist()):
        copyfile(path, os.path.join(output_path, "{}.jpg".format(i)))



def main():
    args = get_args_parser().parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    df1 = pd.read_csv(args.paths2scores1, header=None)
    df2 = pd.read_csv(args.paths2scores2, header=None)
    cond1 = df1[0].isin(df2[0])
    cond2 = df2[0].isin(df1[0])

    df1.drop(df1[cond1].index, inplace=True)
    df2.drop(df2[cond2].index, inplace=True)

    copy_images_of_df(df1, os.path.join(args.output_path, args.name1))
    copy_images_of_df(df2, os.path.join(args.output_path, args.name2))




if __name__=="__main__":
    main()