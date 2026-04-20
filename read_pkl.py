import pickle
import argparse
import glob
import os


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def format_ci(ci):
    if ci is None:
        return "None"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def main():
    parser = argparse.ArgumentParser(description="Print per-skin-tone AUC with 95% CI")
    parser.add_argument("--dir", type=str, help="Directory with .pkl files")
    parser.add_argument("--files", nargs="+", help="List of .pkl files")
    args = parser.parse_args()

    # Collect files
    paths = []
    if args.dir:
        paths.extend(glob.glob(os.path.join(args.dir, "*.pkl")))
    if args.files:
        paths.extend(args.files)

    if not paths:
        print("No files found.")
        return

    for path in sorted(paths):
        data = load_pkl(path)

        tone_auc = data.get("ROC_AUC_by_skin_tone", {})
        tone_ci = data.get("ROC_AUC_by_skin_tone_95CI", {})

        print(f"\n=== {os.path.basename(path)} ===")

        for tone in sorted(tone_auc.keys(), key=lambda x: int(x)):
            auc_val = tone_auc[tone]
            ci_val = tone_ci.get(tone)

            if auc_val is None:
                print(f"Skin tone {tone}: AUC=None, CI=None")
            else:
                print(f"Skin tone {tone}: AUC={auc_val:.4f}, CI={format_ci(ci_val)}")


if __name__ == "__main__":
    main()