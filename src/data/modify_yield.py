import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=str,
        help="Path to yield data, one value per line."
    )

    args = parser.parse_args()

    with open(args.source) as rf:
        for line in rf:
            num = line.strip()
            if float(num) >= 100:
                num = 99.99
            else:
                num = float(num)
            print('{:05.2f}'.format(num))

if __name__ == "__main__":
    main()
