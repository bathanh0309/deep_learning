import matplotlib.pyplot as plt
import numpy as np

# 1. Dữ liệu cập nhật từ log training của bạn (23 epochs)
epochs = np.arange(1, 24)

train_acc = [0.0462, 0.1339, 0.1922, 0.2434, 0.2818, 0.3199, 0.3510, 0.3780, 0.3986, 0.4166,
             0.4420, 0.4628, 0.4736, 0.4861, 0.5013, 0.5177, 0.5259, 0.5342, 0.5483, 0.5756,
             0.5957, 0.6113, 0.6141]

train_loss = [4.7539, 3.6936, 3.3279, 3.0502, 2.8595, 2.6549, 2.5101, 2.3821, 2.2717, 2.1702,
              2.0706, 1.9864, 1.9197, 1.8734, 1.8112, 1.7291, 1.6991, 1.6627, 1.5944, 1.4919,
              1.3995, 1.3384, 1.3320]

val_acc = [0.1462, 0.1862, 0.2894, 0.2910, 0.1940, 0.3760, 0.4034, 0.4108, 0.4206, 0.4112,
           0.4432, 0.4666, 0.4852, 0.4516, 0.4886, 0.4996, 0.4956, 0.4898, 0.4532, 0.5108,
           0.5252, 0.5258, 0.5210]

val_loss = [3.6977, 3.4121, 2.8285, 2.8044, 3.5964, 2.3740, 2.2489, 2.2492, 2.1819, 2.2306,
            2.0865, 1.9660, 1.9278, 2.0903, 1.8823, 1.8992, 1.8783, 1.9543, 2.1269, 1.8398,
            1.8210, 1.8505, 1.8655]

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

# --- ĐÁNH DẤU EPOCH TỐT NHẤT MỚI ---
best_epoch = 22
ax1.axvline(x=best_epoch, color='green', linestyle=':', linewidth=2, label=f'Best Epoch: {best_epoch}')

# 3. Hoàn thiện biểu đồ
plt.title('Training & Validation Metrics for Big Kernel (9x9)', fontsize=16, fontweight='bold')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

fig.tight_layout()
plt.show()