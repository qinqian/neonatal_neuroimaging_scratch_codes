import argparse
from plot import plot_with_mask


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mask')
    p.add_argument('--mask2')
    p.add_argument('--t2flair')
    p.add_argument('--name')
    args = p.parse_args()
    plot_with_mask(args.t2flair, args.name, args.mask, args.mask2)
