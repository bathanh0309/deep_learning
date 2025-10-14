# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
import warnings
from inspect import signature
import plotly.colors as pc
import re

def get_100_colors():
    seq = (pc.qualitative.Dark24 + pc.qualitative.Light24 + pc.qualitative.Alphabet +
           pc.qualitative.Set3 + pc.qualitative.Pastel + pc.qualitative.Safe +
           pc.qualitative.Vivid + pc.qualitative.Bold)
    seen, palette = set(), []
    for c in seq:
        if c not in seen:
            palette.append(c); seen.add(c)
    if len(palette) < 100:
        need = 100 - len(palette)
        palette += pc.sample_colorscale('Viridis', [i/max(need-1, 1) for i in range(need)])
    return palette[:100]

def parse_color_to_rgb(color_str):
    """Chuyển màu từ bất kỳ format nào về RGB tuple (0-1)"""
    if color_str.startswith('#'):
        # Hex format
        color_str = color_str.lstrip('#')
        return tuple(int(color_str[i:i+2], 16)/255.0 for i in (0, 2, 4))
    elif color_str.startswith('rgb'):
        # RGB/RGBA format: rgb(r,g,b) or rgba(r,g,b,a)
        nums = re.findall(r'\d+\.?\d*', color_str)
        r, g, b = float(nums[0]), float(nums[1]), float(nums[2])
        # Nếu giá trị > 1, giả sử là 0-255
        if r > 1 or g > 1 or b > 1:
            return (r/255.0, g/255.0, b/255.0)
        return (r, g, b)
    else:
        # Fallback
        return (0.5, 0.5, 0.5)

warnings.filterwarnings('ignore')

class CIFAR100Visualizer:
    def _fine_to_coarse_map(self):
        """Trả về dict: fine_id -> coarse_id"""
        fine_arr = np.array(self.train_data[b'fine_labels'])
        coarse_arr = np.array(self.train_data[b'coarse_labels'])
        mapping = {}
        for i in range(len(self.fine_labels)):
            idx = np.where(fine_arr == i)[0][0]
            mapping[i] = int(coarse_arr[idx])
        return mapping
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.meta = None
        self.train_data = None
        self.test_data = None
        self.load_data()
        
    def unpickle(self, file):
        """Load CIFAR-100 pickle file"""
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def load_data(self):
        """Load toàn bộ dataset CIFAR-100"""
        print("Loading CIFAR-100 dataset...")
        
        # Load meta data
        meta_path = self.data_path / 'meta'
        self.meta = self.unpickle(meta_path)
        self.fine_labels = [label.decode('utf-8') for label in self.meta[b'fine_label_names']]
        self.coarse_labels = [label.decode('utf-8') for label in self.meta[b'coarse_label_names']]
        
        # Load train data
        train_path = self.data_path / 'train'
        self.train_data = self.unpickle(train_path)
        
        # Load test data  
        test_path = self.data_path / 'test'
        self.test_data = self.unpickle(test_path)
        
        print(f"Loaded {len(self.train_data[b'data'])} training images")
        print(f"Loaded {len(self.test_data[b'data'])} test images")
        print(f"Total classes: {len(self.fine_labels)}")
        
    def prepare_images(self, num_samples=1000):
        """Chuẩn bị images và labels cho visualization"""
        indices = np.random.choice(len(self.train_data[b'data']), num_samples, replace=False)
        
        images = []
        labels = []
        coarse_labels = []
        
        for idx in indices:
            img = self.train_data[b'data'][idx].reshape(3, 32, 32).transpose(1, 2, 0)
            images.append(img)
            labels.append(self.train_data[b'fine_labels'][idx])
            coarse_labels.append(self.train_data[b'coarse_labels'][idx])
            
        return np.array(images), np.array(labels), np.array(coarse_labels)
    
    def _get_tsne_kwargs(self):
        """Lấy kwargs phù hợp với phiên bản scikit-learn"""
        sig = signature(TSNE.__init__)
        kwargs = {
            'n_components': 3,
            'perplexity': 30,
            'random_state': 42
        }
        
        if 'max_iter' in sig.parameters:
            kwargs['max_iter'] = 1000
        elif 'n_iter' in sig.parameters:
            kwargs['n_iter'] = 1000
            
        if 'learning_rate' in sig.parameters:
            try:
                kwargs['learning_rate'] = 'auto'
            except:
                kwargs['learning_rate'] = 200.0
        
        if 'init' in sig.parameters:
            kwargs['init'] = 'pca'
            
        return kwargs
    
    def create_3d_tsne_animation(self, num_samples=2000):
        print("\n Creating 3D t-SNE Interactive Visualization (DARK THEME)...")

        images, labels, coarse_labels = self.prepare_images(num_samples)
        X = images.reshape(num_samples, -1)

        print(" Running PCA...")
        X_pca = PCA(n_components=50).fit_transform(X)

        print(" Running t-SNE (this may take a minute)...")
        tsne_kwargs = self._get_tsne_kwargs()
        X_tsne = TSNE(**tsne_kwargs).fit_transform(X_pca)

        df = pd.DataFrame({
            'x': X_tsne[:, 0],
            'y': X_tsne[:, 1],
            'z': X_tsne[:, 2],
            'fine_id': labels,
            'coarse_id': coarse_labels
        })
        df['fine_label'] = df['fine_id'].map(lambda i: self.fine_labels[int(i)])
        df['coarse_label'] = df['coarse_id'].map(lambda i: self.coarse_labels[int(i)])

        palette = get_100_colors()
        fine_colors = {self.fine_labels[i]: palette[i] for i in range(len(self.fine_labels))}
        f2c = self._fine_to_coarse_map()
        f2c_name = {i: self.coarse_labels[cid] for i, cid in f2c.items()}

        fig = go.Figure()

        for fine_name in sorted(df['fine_label'].unique()):
            sdf = df[df['fine_label'] == fine_name]
            fine_id = self.fine_labels.index(fine_name)
            coarse_name = f2c_name[fine_id]

            fig.add_trace(go.Scatter3d(
                x=sdf['x'], y=sdf['y'], z=sdf['z'],
                mode='markers',
                name=fine_name,
                legendgroup=coarse_name,
                marker=dict(
                    size=5, color=fine_colors[fine_name],
                    opacity=0.85, line=dict(width=0.5, color='white')
                ),
                text=[f"{fine_name} • {coarse_name}"] * len(sdf),
                hovertemplate="<b>%{text}</b><br>x:%{x:.2f} y:%{y:.2f} z:%{z:.2f}<extra></extra>"
            ))

        # DARK THEME
        fig.update_layout(
            title=dict(text=' CIFAR-100 3D t-SNE Visualization', x=0.5, font=dict(color='white', size=24)),
            scene=dict(
                xaxis=dict(backgroundcolor="rgb(10,10,15)", gridcolor="rgb(50,50,60)", showbackground=True),
                yaxis=dict(backgroundcolor="rgb(10,10,15)", gridcolor="rgb(50,50,60)", showbackground=True),
                zaxis=dict(backgroundcolor="rgb(10,10,15)", gridcolor="rgb(50,50,60)", showbackground=True),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor="rgb(0,0,0)",
            plot_bgcolor="rgb(0,0,0)",
            font=dict(color="white"),
            showlegend=True,
            legend=dict(
                title="Fine Classes (click to toggle)",
                yanchor="top", y=1, xanchor="left", x=1.02,
                bgcolor="rgba(20,20,30,0.8)", bordercolor="gray", borderwidth=1,
                itemsizing="constant", font=dict(size=10, color='white')
            ),
            margin=dict(r=270),
            width=1200, height=800
        )

        frames = []
        for deg in range(0, 360, 3):
            frames.append(go.Frame(layout=dict(scene_camera=dict(
                eye=dict(x=2*np.cos(np.radians(deg)), y=2*np.sin(np.radians(deg)), z=1.5)
            ))))
        fig.frames = frames
        fig.update_layout(updatemenus=[dict(
            type="buttons", showactive=False, x=0.1, y=0.95,
            buttons=[
                dict(label="▶ Play", method="animate",
                    args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]),
                dict(label="⏸ Pause", method="animate",
                    args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
            ]
        )])

        fig.write_html("cifar100_3d_visualization.html")
        fig.show()
        print(" 3D Visualization saved as 'cifar100_3d_visualization.html'")

    def create_image_grid_animation(self, num_frames=100, images_per_frame=64):
        """Tạo video animation với nền đen"""
        print("\n Creating Image Grid Animation (DARK THEME)...")
        
        frames = []
        
        for frame_idx in tqdm(range(num_frames), desc="Creating frames"):
            indices = np.random.choice(len(self.train_data[b'data']), images_per_frame, replace=False)
            
            grid_size = int(np.sqrt(images_per_frame))
            # NỀN ĐEN
            grid = np.zeros((grid_size * 32, grid_size * 32, 3), dtype=np.uint8)
            
            for i, idx in enumerate(indices):
                row = i // grid_size
                col = i % grid_size
                
                img = self.train_data[b'data'][idx].reshape(3, 32, 32).transpose(1, 2, 0)
                grid[row*32:(row+1)*32, col*32:(col+1)*32] = img
            
            grid = cv2.resize(grid, (512, 512), interpolation=cv2.INTER_CUBIC)
            
            label_idx = self.train_data[b'fine_labels'][indices[0]]
            label = self.fine_labels[label_idx]
            cv2.putText(grid, f'CIFAR-100: {label}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            frames.append(grid)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('cifar100_grid_animation.mp4', fourcc, 15.0, (512, 512))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(" Animation saved as 'cifar100_grid_animation.mp4'")
        
    def create_class_distribution_sunburst(self):
        """Sunburst với dark theme"""
        print("\n Creating Interactive Sunburst Chart (DARK THEME)...")
        
        fine_counts = np.bincount(self.train_data[b'fine_labels'])
        coarse_counts = np.bincount(self.train_data[b'coarse_labels'])
        
        data = []
        data.append(dict(ids='CIFAR-100', labels='CIFAR-100', parents='', values=sum(fine_counts)))
        
        for i, coarse_label in enumerate(self.coarse_labels):
            data.append(dict(
                ids=coarse_label,
                labels=coarse_label,
                parents='CIFAR-100',
                values=coarse_counts[i]
            ))
        
        for i, fine_label in enumerate(self.fine_labels):
            coarse_idx = self.train_data[b'coarse_labels'][self.train_data[b'fine_labels'].index(i)]
            parent = self.coarse_labels[coarse_idx]
            data.append(dict(
                ids=f"{parent}-{fine_label}",
                labels=fine_label,
                parents=parent,
                values=fine_counts[i]
            ))
        
        df = pd.DataFrame(data)
        
        fig = go.Figure(go.Sunburst(
            ids=df['ids'],
            labels=df['labels'],
            parents=df['parents'],
            values=df['values'],
            branchvalues="total",
            marker=dict(
                colorscale='Viridis',
                cmid=250
            ),
            hovertemplate='<b>%{label}</b><br>Samples: %{value}<extra></extra>',
            textfont=dict(size=12, color='white')
        ))
        
        # DARK THEME
        fig.update_layout(
            title={
                'text': ' CIFAR-100 Hierarchical Class Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': 'white'}
            },
            paper_bgcolor='rgb(0,0,0)',
            plot_bgcolor='rgb(0,0,0)',
            font=dict(color='white'),
            width=900,
            height=900
        )
        
        fig.write_html("cifar100_sunburst.html")
        fig.show()
        print(" Sunburst chart saved as 'cifar100_sunburst.html'")
    
    def create_tsne_orbit_gif_with_legend(self, num_samples=2000, outfile="cifar100_3d_orbit_labeled.gif"):
        """GIF xoay 360° với nền đen"""
        print("\n🎞️ Creating rotating GIF with legend (DARK THEME)...")
        
        images, labels, _ = self.prepare_images(num_samples)
        X = images.reshape(num_samples, -1)
        X_pca = PCA(n_components=50).fit_transform(X)
        
        tsne_kwargs = self._get_tsne_kwargs()
        X_tsne = TSNE(**tsne_kwargs).fit_transform(X_pca)

        palette = get_100_colors()
        colors = np.array([palette[l] for l in labels])
        
        # FIX: Parse colors correctly
        rgb = np.array([parse_color_to_rgb(c) for c in colors])

        # DARK THEME
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 8), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        ax.set_facecolor('black')
        
        sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], s=8, c=rgb, depthshade=False)
        ax.set_axis_off()
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=parse_color_to_rgb(palette[i]), label=self.fine_labels[i]) 
                   for i in range(100)]
        lgd = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), 
                       borderaxespad=0., ncol=1, fontsize=6, facecolor='black', edgecolor='white')
        plt.tight_layout(rect=[0, 0, 0.78, 1])

        def init(): return fig,

        def update(angle):
            ax.view_init(elev=20, azim=angle)
            return fig,

        anim = FuncAnimation(fig, update, init_func=init, frames=range(0, 360, 3), interval=50, blit=True)
        anim.save(outfile, writer='pillow', fps=20)
        plt.close(fig)
        print(f" Saved GIF to {outfile}")

def main():
    # Đường dẫn đến CIFAR-100 dataset
    DATA_PATH = r"C:\Users\LENOVO\Desktop\DeepLearning\cifar-100-python"
    
    # Khởi tạo visualizer
    viz = CIFAR100Visualizer(DATA_PATH)
    
    print("\n" + "="*60)
    print(" CIFAR-100 ADVANCED VISUALIZATION SUITE (DARK THEME)")
    print("="*60)
    
    while True:
        print("\n Select visualization type:")
        print("1. 3D t-SNE Interactive Visualization")
        print("2. Image Grid Animation Video")
        print("3. Hierarchical Sunburst Chart")
        print("4. 3D Orbit GIF with Legend")
        print("5. Run All Visualizations")
        print("0. Exit")
        
        choice = input("\n✨ Your choice: ")
        
        if choice == '1':
            viz.create_3d_tsne_animation()
        elif choice == '2':
            viz.create_image_grid_animation()
        elif choice == '3':
            viz.create_class_distribution_sunburst()
        elif choice == '4':
            viz.create_tsne_orbit_gif_with_legend()
        elif choice == '5':
            viz.create_3d_tsne_animation()
            viz.create_image_grid_animation()
            viz.create_class_distribution_sunburst()
            viz.create_tsne_orbit_gif_with_legend()
            print("\n All visualizations completed!")
        elif choice == '0':
            print(" Goodbye!")
            break
        else:
            print(" Invalid choice!")

if __name__ == "__main__":
    main()