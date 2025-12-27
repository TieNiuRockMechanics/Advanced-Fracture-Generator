import os
import shutil
import sys

# 获取当前脚本所在的目录 (即 FD-Advanced3D-Release 文件夹)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置源路径 (相对于本脚本的上一级目录找到源代码)
# 假设结构是:
# shuixin-master/
#   FD-Advanced3D-Release/ (本脚本在此)
#   分形维数模拟裂缝/
#       FD-Advanced3D/
#       FractureCore/

# 向上两级找到 shuixin-master 根目录
ROOT_DIR = os.path.dirname(CURRENT_DIR) 

SOURCE_DIR = os.path.join(ROOT_DIR, "分形维数模拟裂缝", "FD-Advanced3D")
CORE_DIR = os.path.join(ROOT_DIR, "分形维数模拟裂缝", "FractureCore")
RELEASE_DIR = CURRENT_DIR

def update_repo():
    print("=========================================")
    print("   开始同步代码到当前文件夹...")
    print("=========================================")

    if not os.path.exists(SOURCE_DIR):
        print(f"错误: 找不到源目录 {SOURCE_DIR}")
        print("请确认文件夹结构位置是否正确。")
        return

    # 1. 复制主程序代码 (.py)
    print(f"1. 正在从主项目同步 .py 文件...")
    count = 0
    for filename in os.listdir(SOURCE_DIR):
        # 排除本脚本自己，防止被覆盖（虽然源目录里应该没有这个脚本）
        if filename.endswith(".py") and filename != "test_import.py" and filename != "update_release_repo.py":
            src_file = os.path.join(SOURCE_DIR, filename)
            dst_file = os.path.join(RELEASE_DIR, filename)
            shutil.copy2(src_file, dst_file)
            count += 1
    print(f"   - 已更新 {count} 个文件")
    
    # 2. 复制/更新 translations 文件夹
    print(f"2. 同步翻译文件...")
    trans_src = os.path.join(SOURCE_DIR, "translations")
    trans_dst = os.path.join(RELEASE_DIR, "translations")
    if os.path.exists(trans_dst):
        shutil.rmtree(trans_dst)
    if os.path.exists(trans_src):
        shutil.copytree(trans_src, trans_dst)
        print("   - translations 文件夹已更新")

    # 3. 复制/更新 FractureCore
    print(f"3. 同步 FractureCore 核心库...")
    core_dst = os.path.join(RELEASE_DIR, "FractureCore")
    if os.path.exists(core_dst):
        shutil.rmtree(core_dst) # 先删除旧的，确保完全同步
    shutil.copytree(CORE_DIR, core_dst)
    print("   - FractureCore 已更新")

    # 4. 重新写入“独立版”专用的导入垫片 (Shim)
    print(f"4. 修正独立运行所需的导入路径...")
    
    shim_fracture = """# fracture_model.py
# 独立发布版专用 Shim
# 自动生成的代码，请勿手动修改。请在主项目中修改源码。

import sys
import os

try:
    from FractureCore.fracture_model import FractalBasedFractureGenerator, EllipticalFracture
except ImportError:
    raise ImportError("Could not import FractureCore. Please ensure the 'FractureCore' directory exists in the current directory.")
"""
    with open(os.path.join(RELEASE_DIR, "fracture_model.py"), "w", encoding="utf-8") as f:
        f.write(shim_fracture)

    shim_random = """# random_manager.py
# 独立发布版专用 Shim
# 自动生成的代码，请勿手动修改。请在主项目中修改源码。

import sys
import os

try:
    from FractureCore.random_manager import RandomStateManager
except ImportError:
    raise ImportError("Could not import FractureCore. Please ensure the 'FractureCore' directory exists in the current directory.")
"""
    with open(os.path.join(RELEASE_DIR, "random_manager.py"), "w", encoding="utf-8") as f:
        f.write(shim_random)

    print("=========================================")
    print("   同步完成！")
    print("=========================================")
    print("请执行以下 Git 命令推送更新：")
    print("git status")
    print("git add .")
    print("git commit -m 'Sync latest changes'")
    print("git push")

if __name__ == "__main__":
    update_repo()
