import os
import json

# 文件夹路径
folder_path = 'tasks/stone_tools'

# 获取文件夹中的所有文件
files = os.listdir(folder_path)

# 遍历文件夹中的每一个文件
for file in files:
    # 构建每一个文件的完整路径
    file_path = os.path.join(folder_path, file)

    # 获取原文件名（不包括扩展名）
    filename_without_ext = os.path.splitext(file)[0]
    
    # 新的文件名为“原来的名字_without_task_decomposer”
    new_filename = filename_without_ext + "_without_task_decomposer.json"
    new_file_path = os.path.join(folder_path, new_filename)
    
    # 读取每一个文件
    with open(file_path, 'r') as f:
        # 加载json文件
        data = json.load(f)
        
        # 用于存储所有tips的列表
        all_tips = []
        
        # 读取第一个json字典的非tips内容
        first_dict = data[0]
        first_dict_without_tips = {key: value for key, value in first_dict.items() if key != 'tips'}
        
        # 收集所有的tips
        for item in data:
            all_tips.append(item['tips'])
            
        # 创建一个新的json字典，这个字典中只有一个元素，元素的内容是第一个json字典的非tips内容，新的tips是所有的tips用\n拼接而成
        new_dict = first_dict_without_tips
        new_dict['tips'] = '\n'.join(all_tips)
    
    # 将新的json字典写入新的文件中
    with open(new_file_path, 'w') as f:
        json.dump([new_dict], f)