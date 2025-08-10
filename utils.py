import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

def setup_matplotlib_font():
    """
    A robust function to set the global Chinese font for matplotlib.
    """
    print("Setting up matplotlib Chinese font...")
    
    font_path = None
    if os.name == 'nt': # For Windows
        font_dir = 'C:/Windows/Fonts'
        font_names_map = {
            'SimHei': 'simhei.ttf', 'Microsoft YaHei': 'msyh.ttc', 'SimSun': 'simsun.ttc'
        }
        for name, filename in font_names_map.items():
            path = os.path.join(font_dir, filename)
            if os.path.exists(path):
                font_path = path
                break
    else: # For macOS/Linux (can be expanded)
        font_paths_to_check = ['/System/Library/Fonts/PingFang.ttc', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']
        for path in font_paths_to_check:
            if os.path.exists(path):
                font_path = path
                break
    
    if font_path:
        try:
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Matplotlib Chinese font '{font_prop.get_name()}' set successfully.")
        except Exception as e:
            print(f"Error setting matplotlib font: {e}")
    else:
        print("Warning: Could not find a suitable Chinese font. Chinese characters may not display correctly.")