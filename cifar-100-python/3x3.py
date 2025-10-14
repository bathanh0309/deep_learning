import matplotlib.pyplot as plt
import numpy as np

# 1. Dữ liệu trích xuất từ log training của bạn (26 epochs)
epochs = np.arange(1, 27)

train_acc = [0.4134, 0.4372, 0.4601, 0.4728, 0.4816, 0.5009, 0.5105, 0.5237, 0.5369, 0.5485,
             0.5544, 0.5664, 0.5734, 0.5878, 0.5937, 0.6263, 0.6447, 0.6536, 0.6640, 0.6716,
             0.6771, 0.6974, 0.7070, 0.7110, 0.7185, 0.7213]

train_loss = [2.2040, 2.0911, 2.0048, 1.9385, 1.8847, 1.8155, 1.7611, 1.7090, 1.6584, 1.6079,
              1.5775, 1.5271, 1.5036, 1.4474, 1.4102, 1.2615, 1.2088, 1.1711, 1.1303, 1.1036,
              1.0755, 1.0023, 0.9623, 0.9513, 0.9301, 0.9132]

val_acc = [0.4470, 0.4678, 0.4674, 0.4728, 0.4634, 0.4736, 0.4812, 0.4898, 0.4894, 0.5048,
           0.4944, 0.5094, 0.5028, 0.5058, 0.5048, 0.5306, 0.5332, 0.5272, 0.5304, 0.5306,
           0.5326, 0.5378, 0.5332, 0.5394, 0.5370, 0.5356]

val_loss = [2.0675, 2.0039, 1.9953, 1.9775, 2.0316, 2.0286, 1.9611, 1.9540, 1.9392, 1.8659,
            1.9490, 1.8984, 1.8877, 1.9025, 1.9136, 1.8486, 1.8739, 1.8840, 1.8957, 1.8893,
            1.9087, 1.8846, 1.9180, 1.9245, 1.9338, 1.9617]

# 2. Vẽ biểu đồ với hai trục Y
fig, ax1 = plt.subplots(figsize=(12, 7))

# Trục Y bên trái cho Accuracy
color = 'tab:blue'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', color=color, fontsize=12)
ax1.plot(epochs, train_acc, color=color, linestyle='-', marker='o', markersize=4, label='Training Accuracy')
ax1.plot(epochs, val_acc, color=color, linestyle='--', marker='x', markersize=4, label='Validation Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

# Trục Y bên phải cho Loss
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Loss', color=color, fontsize=12)
ax2.plot(epochs, train_loss, color=color, linestyle='-', marker='o', markersize=4, label='Training Loss')
ax2.plot(epochs, val_loss, color=color, linestyle='--', marker='x', markersize=4, label='Validation Loss')
ax2.tick_params(axis='y', labelcolor=color)

# --- ĐÁNH DẤU EPOCH TỐT NHẤT ---
best_epoch = 16
ax1.axvline(x=best_epoch, color='green', linestyle=':', linewidth=2, label=f'Best Epoch: {best_epoch}')

# 3. Hoàn thiện biểu đồ
plt.title('Training & Validation Metrics', fontsize=16, fontweight='bold')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

fig.tight_layout()
plt.show()