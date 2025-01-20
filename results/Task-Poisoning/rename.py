import os
import re

# 目标目录
log_dir = "."

# 遍历目录中的所有文件
for filename in os.listdir(log_dir):
    # 检查是否是 JSON 文件
    if filename.endswith(".json"):
        # 使用正则表达式去掉时间部分
        new_filename = re.sub(r"^\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-", "", filename)
        
        # 如果文件名确实发生了变化，则重命名
        if new_filename != filename:
            old_path = os.path.join(log_dir, filename)
            new_path = os.path.join(log_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")