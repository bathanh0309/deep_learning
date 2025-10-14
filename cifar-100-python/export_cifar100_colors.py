# export_cifar100_colors.py
import os, csv, pickle, argparse
import plotly.colors as pc

def get_100_colors():
    seq = (pc.qualitative.Dark24 + pc.qualitative.Light24 + pc.qualitative.Alphabet +
           pc.qualitative.Set3 + pc.qualitative.Pastel + pc.qualitative.Safe +
           pc.qualitative.Vivid + pc.qualitative.Bold)
    seen, palette = set(), []
    for c in seq:
        if c not in seen:
            if isinstance(c, str):
                if c.startswith("#"):  # hex rồi
                    palette.append(c)
                elif c.startswith("rgb"):  # dạng "rgb(r,g,b)"
                    nums = c.replace("rgb(", "").replace(")", "").split(",")
                    r,g,b = [int(v) for v in nums]
                    palette.append(f"#{r:02x}{g:02x}{b:02x}")
                else:
                    try:
                        from matplotlib.colors import to_hex
                        palette.append(to_hex(c))
                    except Exception:
                        pass
            seen.add(c)
    # bổ sung nếu <100
    if len(palette) < 100:
        need = 100 - len(palette)
        extra = pc.sample_colorscale("Viridis", [i/max(need-1,1) for i in range(need)])
        for e in extra:
            if isinstance(e, str) and e.startswith("rgb"):
                nums = e.replace("rgb(", "").replace(")", "").split(",")
                r,g,b = [int(float(v)) for v in nums]
                palette.append(f"#{r:02x}{g:02x}{b:02x}")
            elif isinstance(e, str) and e.startswith("#"):
                palette.append(e)
            else:
                from matplotlib.colors import to_hex
                palette.append(to_hex(e))
    return palette[:100]

def load_meta_and_map(root):
    meta_p = os.path.join(root, "cifar-100-python", "meta")
    train_p = os.path.join(root, "cifar-100-python", "train")
    with open(meta_p, "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    with open(train_p, "rb") as f:
        train = pickle.load(f, encoding="latin1")

    fine_names   = meta["fine_label_names"]      # len=100
    coarse_names = meta["coarse_label_names"]    # len=20
    fine_labels   = train["fine_labels"]         # per-sample
    coarse_labels = train["coarse_labels"]       # per-sample

    # map fine_id -> coarse_id (lay theo mau trong file train, la duy nhat)
    fine2coarse = {}
    for fid in range(100):
        idx = fine_labels.index(fid)  # vi tri dau tien
        fine2coarse[fid] = int(coarse_labels[idx])

    return fine_names, coarse_names, fine2coarse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--out", type=str, default="cifar100_class_colors.csv")
    args = ap.parse_args()

    fine_names, coarse_names, fine2coarse = load_meta_and_map(args.data_root)
    colors = get_100_colors()  # 100 mau HEX

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id","fine_label","coarse_id","coarse_label","color_hex"])
        for fid, name in enumerate(fine_names):
            cid = fine2coarse[fid]
            w.writerow([fid, name, cid, coarse_names[cid], colors[fid]])

    print("Saved:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
# python export_cifar100_colors.py --data_root "."
# -> tao file cifar100_class_colors.csv
