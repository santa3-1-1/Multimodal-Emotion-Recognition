import argparse
from predict import predict_from_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--text', type=str)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    if args.mode == 'predict':
        predict_from_args(args)
    else:
        print("暂不支持该模式")


if __name__ == "__main__":
    main()
