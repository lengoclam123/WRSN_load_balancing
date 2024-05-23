import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip
from IPython.display import Video, display

# Tạo danh sách để lưu tên các tệp hình ảnh
image_files = []

# Tạo và lưu các plot
for i in range(10):
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x + i * 0.1)
    
    plt.figure()
    plt.plot(x, y)
    filename = f'plot_{i}.png'
    plt.savefig(filename)
    image_files.append(filename)
    plt.close()

# Đọc các hình ảnh và tạo video
clip = ImageSequenceClip(image_files, fps=2)  # fps là số khung hình trên giây

# Lưu video
video_file = 'plots_video.mp4'
clip.write_videofile(video_file, codec='libx264')

# Hiển thị video trong Jupyter Notebook
display(Video(video_file))
