import os
import json

def read_files_to_matrix(file_paths):
    matrix = []
    
    # 读取所有文件的行数，假设所有文件的行数相同
    num_lines = None
    all_lines = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if num_lines is None:
                num_lines = len(lines)
            elif num_lines != len(lines):
                raise ValueError(f"文件 {file_path} 的行数与其他文件不匹配。")
            all_lines.append(lines)

    for i in range(num_lines):
        row = [int(all_lines[j][i].strip()) for j in range(len(file_paths))]
        matrix.append(row)

    return matrix

# 假设你的txt文件路径保存在一个列表中
file_paths = ['result_0.txt',
'result_1.txt',
'result_2.txt',
'result_3.txt',
'result_4.txt']

# 读取文件并转换为矩阵
matrix = read_files_to_matrix(file_paths)

# 将矩阵保存到一个JSON文件中
output_file_path = '/stacking/train.json'
with open(output_file_path, 'w') as json_file:
    json.dump(matrix, json_file)

print(f"矩阵已保存到 {output_file_path}")
