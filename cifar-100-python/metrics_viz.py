# metrics_viz.py
import os, glob, argparse, re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # ép kiểu
    for c in ["epoch","train_loss","train_acc","test_loss","test_acc","lr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_from_tb(logdir):
    # lấy event file mới nhất
    runs = sorted(glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True))
    if not runs:
        raise FileNotFoundError("Không tìm thấy file TensorBoard trong logdir")
    ev = EventAccumulator(os.path.dirname(runs[-1]))
    ev.Reload()
    def get_series(tag):
        if tag in ev.Scalars(tag):
            pass
        sc = ev.Scalars(tag)
        if not sc: return None
        # step là epoch theo cách bạn add_scalar(tag, value, epoch)
        return pd.DataFrame({"epoch":[s.step for s in sc], tag:[s.value for s in sc]})
    tags = ["train/loss","train/acc","val/loss","val/acc","lr"]
    dfs = []
    for t in tags:
        try:
            s = ev.Scalars(t)
        except KeyError:
            s = None
        if s:
            dfs.append(pd.DataFrame({"epoch":[x.step for x in s], t:[x.value for x in s]}))
    if not dfs:
        raise RuntimeError("Không đọc được scalar từ TB; kiểm tra tag add_scalar")
    df = dfs[0]
    for d in dfs[1:]:
        df = pd.merge(df, d, on="epoch", how="outer")
    # đổi tên cột cho thống nhất
    df = df.rename(columns={"val/loss":"test_loss","val/acc":"test_acc","train/loss":"train_loss","train/acc":"train_acc"})
    df = df.sort_values("epoch")
    return df

def make_html(df, out_html="training_dashboard.html"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, subplot_titles=("Accuracy (%)", "Loss"))
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_acc"], name="Train Acc", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["test_acc"],  name="Test Acc",  mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train Loss", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["test_loss"],  name="Test Loss",  mode="lines"), row=2, col=1)

    # vẽ LR (nếu có) ở trục phụ dưới
    if "lr" in df.columns:
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["lr"], name="LR", mode="lines", yaxis="y3"), row=2, col=1)
        fig.update_layout(yaxis3=dict(title="LR", overlaying="y2", side="right", showgrid=False, type="log"))

    # tạo frames để “animate”
    frames = []
    for k in range(1, len(df)+1):
        frames.append(go.Frame(data=[
            go.Scatter(x=df["epoch"][:k], y=df["train_acc"][:k]),
            go.Scatter(x=df["epoch"][:k], y=df["test_acc"][:k]),
            go.Scatter(x=df["epoch"][:k], y=df["train_loss"][:k]),
            go.Scatter(x=df["epoch"][:k], y=df["test_loss"][:k]),
        ]))
    fig.frames = frames
    fig.update_layout(
        title="Training Dashboard (interactive)",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        yaxis2_title="Loss",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[dict(type="buttons", showactive=False,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, {"frame":{"duration":50,"redraw":True},
                                                     "fromcurrent":True,"mode":"immediate"}]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], {"frame":{"duration":0,"redraw":False}}])])]
    )
    fig.write_html(out_html)
    print("Saved HTML:", os.path.abspath(out_html))

def make_gif(df, out_gif="training_anim.gif"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,6), sharex=True)
    ax1.set_ylabel("Acc (%)"); ax2.set_ylabel("Loss"); ax2.set_xlabel("Epoch")

    (l1,) = ax1.plot([], [], label="Train Acc")
    (l2,) = ax1.plot([], [], label="Test Acc")
    (l3,) = ax2.plot([], [], label="Train Loss")
    (l4,) = ax2.plot([], [], label="Test Loss")
    ax1.legend(loc="lower right"); ax2.legend(loc="upper right")

    ax1.set_xlim(df["epoch"].min(), df["epoch"].max())
    ax1.set_ylim(0, max(df["train_acc"].max(), df["test_acc"].max(), 1))
    ax2.set_xlim(df["epoch"].min(), df["epoch"].max())
    ax2.set_ylim(0, max(df["train_loss"].max(), df["test_loss"].max(), 1))

    def update(k):
        x = df["epoch"][:k]
        l1.set_data(x, df["train_acc"][:k])
        l2.set_data(x, df["test_acc"][:k])
        l3.set_data(x, df["train_loss"][:k])
        l4.set_data(x, df["test_loss"][:k])
        ax1.set_title(f"Epoch {int(df['epoch'].iloc[k-1])}")
        return l1, l2, l3, l4

    ani = FuncAnimation(fig, update, frames=len(df), interval=50, blit=True)
    ani.save(out_gif, writer="pillow", fps=20)
    plt.close(fig)
    print("Saved GIF:", os.path.abspath(out_gif))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--tb_logdir", type=str, default="./tb_logs")
    ap.add_argument("--out_html", type=str, default="training_dashboard.html")
    ap.add_argument("--out_gif", type=str, default="training_anim.gif")
    args = ap.parse_args()

    if args.csv and os.path.exists(args.csv):
        df = load_from_csv(args.csv)
    else:
        df = load_from_tb(args.tb_logdir)

    # lọc các cột đủ dữ liệu
    need = ["epoch","train_loss","train_acc","test_loss","test_acc"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"Thiếu cột {c}. Hãy dùng CSV hoặc đảm bảo đã add_scalar cho tag tương ứng.")

    make_html(df, args.out_html)
    make_gif(df, args.out_gif)

if __name__ == "__main__":
    main()
# python metrics_viz.py --csv ./ckpts/metrics.csv --out_html training_dashboard.html --out_gif training_anim.gif
