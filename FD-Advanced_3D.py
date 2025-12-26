

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox,
                             QSplitter, QFrame, QCheckBox, QGridLayout, 
                             QFileDialog, QComboBox, QMessageBox, QTabWidget,
                             QTextEdit, QProgressBar, QScrollArea, QTableWidget,
                             QTableWidgetItem, QHeaderView, QDialog, QAbstractItemView,
                             QSlider, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import os
from datetime import datetime
import traceback
import csv
import math
import pandas as pd
import re
import multiprocessing
import pickle
import time
import traceback
import json
import gc
from random_manager import RandomStateManager
from fracture_model import EllipticalFracture, FractalBasedFractureGenerator
from fractal_analysis import FractalDimension3DCalculator, FractalDimension2DCalculator, FractalAnalysisUtils
from drilling_analyzer import DrillingAnalyzer
from exporters import FractureExporter
from data_manager import SQLiteBackupManager, TempFileManager
from translation_manager import TranslationManager
from PyQt5.QtCore import QCoreApplication
from multiprocessing import Queue, Pool
from collections import deque # <<< 新增：导入双端队列作为我们的缓冲区

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from unfolding_analyzer import UnfoldingAnalysisWidget
from unfolding_analyzer import perform_unfolding_analysis






# 设置中文字体支持
import matplotlib as mpl
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
mpl.rc('font', family='Microsoft YaHei')



class QueueListener(QObject):
    """在一个单独的线程中监听多进程队列，并将结果存入一个线程安全的缓冲区"""

    def __init__(self, queue, buffer):
        super().__init__()
        self.queue = queue
        self.buffer = buffer # 新增: 引用GUI线程中的缓冲区
        self._is_running = True

    def run(self):
        """监听循环"""
        while self._is_running:
            try:
                # 使用带超时的get，避免在停止时永久阻塞
                result = self.queue.get(timeout=0.1) 
                # 关键修改: 不再发射信号，而是将结果放入缓冲区
                self.buffer.append(result) 
            except Exception:
                # 队列为空或发生其他小异常时继续
                continue

    def stop(self):
        self._is_running = False

worker_queue = None
stop_event_worker = None # <<< 新增: 为工作进程创建全局变量

def init_worker(queue, stop_event):
    """每个工作进程启动时调用的初始化函数，用于接收队列和停止事件的引用"""
    global worker_queue, stop_event_worker
    worker_queue = queue
    stop_event_worker = stop_event # <<< 新增: 保存对共享停止事件的引用


class SquareAspectCanvas(FigureCanvas):
    """保持1:1宽高比的画布类"""
    
    def __init__(self, figure):
        super().__init__(figure)
        # 设置大小策略，确保保持1:1比例
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy.setHeightForWidth(True)  # 高度根据宽度调整
        self.setSizePolicy(size_policy)
    
    def heightForWidth(self, width):
        """返回与宽度相等的高度，确保1:1比例"""
        return width
    
    def sizeHint(self):
        """返回建议的大小"""
        return self.size()

class FractalDimensionCalculationThread(QThread):
    """分形维数计算线程"""
    
    # 信号定义
    calculation_finished = pyqtSignal(float, dict, list)
    calculation_failed = pyqtSignal(str)
    progress_updated = pyqtSignal(str)
    
    def __init__(self, fractures, cube_size, iterations=6):
        super().__init__()
        self.fractures = fractures
        self.cube_size = cube_size
        self.iterations = iterations
        self.calculator = FractalDimension3DCalculator(iterations)
    
    def run(self):
        """运行计算"""
        # 定义翻译上下文
        CONTEXT = "FractalDimensionCalculationThread"
        try:
            self.progress_updated.emit(
                QCoreApplication.translate(CONTEXT, "开始计算分形维数...")
            )
            
            # 设置进度回调
            # 注意: progress_callback 传递的字符串可能来自更深层的模块。
            # 如果那些模块也需要国际化，它们也需要进行类似的改造。
            # 此处假设 calculator 内部的 progress_callback 调用传递的是非用户可见的调试信息，
            # 或者该模块的国际化已在别处处理。
            self.calculator.progress_callback = self.progress_updated.emit
            
            calculated_dim, fit_data, levels = self.calculator.calculate_fractal_dimension_3d(
                self.fractures, self.cube_size
            )
            
            if calculated_dim is not None:
                self.calculation_finished.emit(calculated_dim, fit_data, levels)
            else:
                self.calculation_failed.emit(
                    QCoreApplication.translate(CONTEXT, "计算失败：数据不足或无有效数据点")
                )
                
        except Exception as e:
            error_prefix = QCoreApplication.translate(CONTEXT, "计算过程中发生错误: ")
            self.calculation_failed.emit(f"{error_prefix}{str(e)}")


# 文件: FD-Advanced_3D.py

# 文件: FD-Advanced_3D.py

# 请用这个新的、完整的 FractureGenerationThread 类定义，替换您代码中旧的同名类
# 文件: FD-Advanced_3D.py

class FractureGenerationThread(QThread):
    """裂缝生成线程"""
    generation_finished = pyqtSignal(object, list)
    generation_failed = pyqtSignal(str)
    progress_updated = pyqtSignal(str)
    
    def __init__(self, generator, cube_size, fractal_dim, n0, iterations,
                 random_aspect_ratio, aspect_ratio_base, aspect_ratio_variation,
                 is_isotropic, mean_inclination, mean_azimuth, kappa, 
                 use_advanced_model, # <<< 新增参数
                 num_slices):
        super().__init__()
        self.generator = generator
        self.cube_size = cube_size
        self.fractal_dim = fractal_dim
        self.n0 = n0
        self.iterations = iterations
        self.random_aspect_ratio = random_aspect_ratio
        self.aspect_ratio_base = aspect_ratio_base
        self.aspect_ratio_variation = aspect_ratio_variation
        self.is_isotropic = is_isotropic
        self.mean_inclination = mean_inclination
        self.mean_azimuth = mean_azimuth
        self.mean_azimuth = mean_azimuth
        self.kappa = kappa
        self.use_advanced_model = use_advanced_model # <<< 新增
        self.num_slices = num_slices
        
        from threading import Event
        self.stop_event = Event()

    def run(self):
        """运行裂缝生成"""
        CONTEXT = "FractureGenerationThread"
        try:
            self.progress_updated.emit(QCoreApplication.translate(CONTEXT, "开始生成分形裂缝..."))
            
            import time
            rng_manager = RandomStateManager(int(time.time() * 1000))

            fractures_by_level, structured_log = self.generator.generate_fractures(
                cube_size=self.cube_size,
                fractal_dimension=self.fractal_dim,
                n0=self.n0,
                max_iterations=self.iterations,
                rng_manager=rng_manager,
                stop_event=self.stop_event,
                
                # 传递所有新的形态参数
                random_aspect_ratio=self.random_aspect_ratio,
                aspect_ratio_base=self.aspect_ratio_base,
                aspect_ratio_variation=self.aspect_ratio_variation,
                is_isotropic=self.is_isotropic,
                mean_inclination=self.mean_inclination,
                mean_azimuth=self.mean_azimuth,
                kappa=self.kappa,
                use_advanced_model=self.use_advanced_model # <<< 传递参数
            )
            
            if self.stop_event.is_set():
                print("生成线程被用户中止。")
                return

            self.progress_updated.emit(QCoreApplication.translate(CONTEXT, "裂缝生成完成，正在后处理..."))
            if self.stop_event.is_set(): return

            all_fractures = self.generator.get_fractures_up_to_level(self.iterations)
            for fracture in all_fractures:
                if self.stop_event.is_set(): return
                fracture.clip_to_cube(self.cube_size, self.num_slices)

            self.progress_updated.emit(QCoreApplication.translate(CONTEXT, "后处理完成，正在更新显示..."))
            
            if not self.stop_event.is_set():
                self.generation_finished.emit(fractures_by_level, structured_log)
                
        except Exception as e:
            if not self.stop_event.is_set():
                import traceback
                error_prefix = QCoreApplication.translate(CONTEXT, "生成裂缝时发生错误: ")
                error_msg = f"{error_prefix}{str(e)}\n{traceback.format_exc()}"
                self.generation_failed.emit(error_msg)

    def stop(self):
        """向线程发出停止请求"""
        print("FractureGenerationThread 收到停止请求。")
        self.stop_event.set()



# ==================== 新增的核心模拟函数 ====================
# 文件: FD-Advanced_3D.py

# 文件: FD-Advanced_3D.py
# 用此版本完整替换 _run_simulation_task 函数
def _run_simulation_task(task_params):
    """
    执行单个参数组合的完整模拟流程。
    *** 包含协作式停止检查 ***
    """
    # 1. 解包所有任务参数
    cube_size = task_params['cube_size']
    fractal_dim = task_params['fractal_dim']
    n0 = task_params['n0']
    repeat_count = task_params['repeat_count']
    gen_iterations = task_params['gen_iterations']
    calc_iterations = task_params['calc_iterations']
    base_seed = task_params['base_seed']
    num_slices = task_params['num_slices'] # <--- 新增：解包切片数参数
    calc_slice_profile = task_params.get('calc_slice_profile', True) 
    calc_slice_profile = task_params.get('calc_slice_profile', True) 
    calc_center_only = task_params.get('calc_center_only', False) 
    use_advanced_model = task_params.get('use_advanced_model', False) # <<< 新增参数 



    
    # 2. 初始化
    temp_generator = FractalBasedFractureGenerator()
    results_for_this_task = []
    
    # 3. 创建随机管理器
    task_rng_manager = RandomStateManager(task_params['base_seed'])

    # 4. 执行所有重复计算
    for repeat in range(task_params['repeat_count']):
        
        global stop_event_worker
        if stop_event_worker is not None and stop_event_worker.is_set():
            break

        try:
            # a. 调用生成器，并传入所有形态参数
            fractures_by_level, _ = temp_generator.generate_fractures(
                cube_size=task_params['cube_size'], 
                fractal_dimension=task_params['fractal_dim'], 
                n0=task_params['n0'],
                max_iterations=task_params['gen_iterations'],
                rng_manager=task_rng_manager,
                
                # (保留) 传递长短轴参数
                random_aspect_ratio=task_params['random_aspect_ratio'],
                aspect_ratio_base=task_params['aspect_ratio_base'],
                aspect_ratio_variation=task_params['aspect_ratio_variation'],
                
                # (新增) 传递新的产状分布参数
                is_isotropic=task_params['is_isotropic'],
                mean_inclination=task_params['mean_inclination'],
                mean_azimuth=task_params['mean_azimuth'],
                kappa=task_params['kappa'],
                use_advanced_model=task_params.get('use_advanced_model', False), # <<< 传递
                num_slices = task_params['num_slices'] # 确保 num_slices 在 task_params 中

            )
            fractures = temp_generator.get_fractures_up_to_level(task_params['gen_iterations'])
            
            if not fractures:
                raise ValueError("未能成功生成任何裂缝。")

            # b. 后处理
            for fracture in fractures:
                fracture.clip_to_cube(task_params['cube_size'], num_slices) # <--- 传递参数
            
            # c. 计算3D分形维数
            calculator_3d = FractalDimension3DCalculator(calc_iterations)
            calculated_3d_dim, fit_3d_data, _ = calculator_3d.calculate_fractal_dimension_3d(
                fractures, cube_size
            )
            
            # d. 计算2D分形维数 (现在返回一个字典)
            avg_2d_data = FractalAnalysisUtils.calculate_average_2d_dimension(
                fractures, cube_size, calc_iterations, gen_iterations, num_slices,
                calc_slice_profile=calc_slice_profile,
                calc_center_only=calc_center_only # <--- 传递
            )


            # --- e.【核心新增】计算表面展开分形维数 ---
            unfolding_data = perform_unfolding_analysis(fractures, cube_size, gen_iterations=gen_iterations)
            
            # f. 封装成功结果 (增加新字段)
            result = {
                'cube_size': float(cube_size), 'theoretical_dim': float(fractal_dim), 'theoretical_n0': float(n0),
                'actual_3d_dim': float(calculated_3d_dim) if calculated_3d_dim is not None else None,
                'actual_3d_n0': float(10 ** fit_3d_data['coeffs'][1]) if fit_3d_data and fit_3d_data.get('coeffs') is not None else None,
                'average_2d_dim_all': float(avg_2d_data.get('all_dim')) if avg_2d_data.get('all_dim') is not None else None,
                'average_2d_n0_all': float(avg_2d_data.get('all_n0')) if avg_2d_data.get('all_n0') is not None else None,
                'average_2d_dim_yoz': float(avg_2d_data.get('yoz_dim')) if avg_2d_data.get('yoz_dim') is not None else None,
                'average_2d_n0_yoz': float(avg_2d_data.get('yoz_n0')) if avg_2d_data.get('yoz_n0') is not None else None,
                'average_2d_dim_xoz': float(avg_2d_data.get('xoz_dim')) if avg_2d_data.get('xoz_dim') is not None else None,
                'average_2d_n0_xoz': float(avg_2d_data.get('xoz_n0')) if avg_2d_data.get('xoz_n0') is not None else None,
                'average_2d_dim_xoy': float(avg_2d_data.get('xoy_dim')) if avg_2d_data.get('xoy_dim') is not None else None,
                'average_2d_n0_xoy': float(avg_2d_data.get('xoy_n0')) if avg_2d_data.get('xoy_n0') is not None else None,

                'slice_profile_dim': avg_2d_data.get('slice_profile_dim'),
                'slice_profile_n0': avg_2d_data.get('slice_profile_n0'),
                # 新增展开分析结果
                'cuboid_unfolding_dim': unfolding_data.get('cuboid_unfolding_dim'),
                'cuboid_unfolding_n0': unfolding_data.get('cuboid_unfolding_n0'),
                'cylinder_unfolding_dim': unfolding_data.get('cylinder_unfolding_dim'),
                'cylinder_unfolding_n0': unfolding_data.get('cylinder_unfolding_n0'),
                'repeat_index': task_params.get('completed_count', 0) + repeat + 1,
                'success': True
            }
        except Exception as e:
            import traceback
            # f. 封装失败结果
            result = {
                'cube_size': cube_size, 'theoretical_dim': fractal_dim, 'theoretical_n0': n0,
                'actual_3d_dim': None, 'actual_3d_n0': None,
                
                # --- 新增：在失败结果中也添加新字段 ---
                'average_2d_dim_all': None, 'average_2d_n0_all': None,
                'average_2d_dim_yoz': None, 'average_2d_n0_yoz': None,
                'average_2d_dim_xoz': None, 'average_2d_n0_xoz': None,
                'average_2d_dim_xoy': None, 'average_2d_n0_xoy': None,
                'slice_profile_dim': None, 'slice_profile_n0': None,
                'cuboid_unfolding_dim': None, 'cuboid_unfolding_n0': None,
                'cylinder_unfolding_dim': None, 'cylinder_unfolding_n0': None,
                
                'repeat_index': task_params.get('completed_count', 0) + repeat + 1,
                'success': False,
                'error': f"{str(e)} \n {traceback.format_exc()}"
            }
        
        results_for_this_task.append(result)
        
        # 内存清理
        if 'fractures_by_level' in locals(): del fractures_by_level
        if 'fractures' in locals(): del fractures
        if 'calculator_3d' in locals(): del calculator_3d
        gc.collect()

    return results_for_this_task
# ==========================================================

def multiprocess_single_calculation_ipc(task_data):
    """
    基于内存队列的多进程参数组合计算函数 (最终版)。
    """
    import gc, sys, traceback
    
    try:
        results_list = _run_simulation_task(task_data)

        global_backup = None
        backup_file_path = task_data.get('backup_file_path')
        if backup_file_path:
            global_backup = SQLiteBackupManager(backup_file_path)
            
        for result in results_list:
            if worker_queue:
                worker_queue.put(result)
            
            if global_backup:
                global_backup.append_result(result)

    except Exception as e:
        error_msg = f"任务在核心引擎中完全失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        
        repeat_count = task_data.get('repeat_count', 1)
        for repeat in range(repeat_count):
            result = {
                'cube_size': task_data.get('cube_size'),
                'theoretical_dim': task_data.get('fractal_dim'),
                'theoretical_n0': task_data.get('n0'),
                'actual_3d_dim': None, 'actual_3d_n0': None,
                'average_2d_dim': None, 'average_2d_n0': None,
                'repeat_index': task_data.get('completed_count', 0) + repeat + 1,
                'success': False,
                'error': error_msg
            }
            if worker_queue:
                worker_queue.put(result)
    finally:
        gc.collect()


def multiprocess_single_calculation(task_data):
    """
    此函数现在调用新的基于内存队列的版本
    """
    return multiprocess_single_calculation_ipc(task_data)





class Study2D3DCalculationThread(QThread):
    """2D-3D关系研究计算线程"""

    calculation_progress = pyqtSignal(int, int, str)  # 进度信号：当前进度，总进度，状态文本
    single_result_ready = pyqtSignal(dict)  # 单个结果完成信号
    calculation_finished = pyqtSignal(int)  # 所有计算完成信号，传递结果数量
    calculation_failed = pyqtSignal(str)  # 计算失败信号
    
# 文件: FD-Advanced_3D.py
# 替换 Study2D3DCalculationThread 的 __init__ 方法

# 文件: FD-Advanced_3D.py (Study2D3DCalculationThread 类)

    def __init__(self, cube_lengths, fractal_dims, n0_values, repeat_count,
                 gen_iterations, calc_iterations, num_slices,
                 calc_slice_profile, # <--- 必须在第8个位置
                 calc_center_only, # <--- 必须添加这个参数
                 random_aspect_ratio, aspect_ratio_base, aspect_ratio_variation,
                 is_isotropic, mean_inclination, mean_azimuth, kappa,
                 use_advanced_model, # <<< 新增参数
                 completed_tasks=None, backup_file_path=None, gui_input_params=None):
        super().__init__()
        self.cube_lengths = cube_lengths
        self.fractal_dims = fractal_dims
        self.n0_values = n0_values
        self.repeat_count = repeat_count
        self.gen_iterations = gen_iterations
        self.calc_iterations = calc_iterations
        self.num_slices = num_slices
        self.calc_slice_profile = calc_slice_profile # 保存开关参数
        
        self.random_aspect_ratio = random_aspect_ratio
        self.aspect_ratio_base = aspect_ratio_base
        self.aspect_ratio_variation = aspect_ratio_variation
        
        self.is_isotropic = is_isotropic
        self.mean_inclination = mean_inclination
        self.mean_azimuth = mean_azimuth
        self.kappa = kappa
        self.use_advanced_model = use_advanced_model
        
        self._stop_requested = False
        self.completed_tasks = completed_tasks or []
        self.backup_file_path = backup_file_path
        self.results = []
        
        self.calculation_params = {
            "cube_lengths": cube_lengths, "fractal_dims": fractal_dims,
            "n0_values": n0_values, "repeat_count": repeat_count,
            "gen_iterations": gen_iterations, "calc_iterations": calc_iterations,
            "num_slices": num_slices,
            "calc_slice_profile": calc_slice_profile,
            "calc_center_only": calc_center_only, # <--- 传递参数 (第9位)
            "random_aspect_ratio": random_aspect_ratio,
            "aspect_ratio_base": aspect_ratio_base,
            "aspect_ratio_variation": aspect_ratio_variation,
            "is_isotropic": is_isotropic,
            "mean_inclination": mean_inclination,
            "mean_azimuth": mean_azimuth,
            "kappa": kappa,
            "use_advanced_model": use_advanced_model,
            "gui_input_params": gui_input_params or {}
        }
    
    def stop_calculation(self):
        """请求停止计算"""
        print("收到停止计算请求...")
        self._stop_requested = True
        
        # 如果有活跃的进程池，尝试优雅地停止它
        if hasattr(self, '_current_pool') and self._current_pool is not None:
            try:
                print("正在停止当前进程池...")
                self._current_pool.close()
                # 不等待join，让run方法中的finally块处理
            except Exception as e:
                print(f"停止进程池时出错: {e}")
    
# 文件: FD-Advanced_3D.py (替换 Study2D3DCalculationThread.run)

# 文件: FD-Advanced_3D.py (替换 Study2D3DCalculationThread.run)

# 文件: FD-Advanced_3D.py
# 替换 Study2D3DCalculationThread 的 run 方法

    def run(self):
        """运行单线程计算 (已修复浮点数比较问题)"""
        try:
            total_combinations = len(self.cube_lengths) * len(self.fractal_dims) * len(self.n0_values)
            total_simulations = total_combinations * self.repeat_count
            current_simulation_count = 0
            task_id_counter = 0

            backup_manager = None
            if self.backup_file_path:
                backup_manager = SQLiteBackupManager(self.backup_file_path, self.calculation_params)

            import math # 导入 math 模块

            for cube_size in self.cube_lengths:
                for fractal_dim in self.fractal_dims:
                    for n0 in self.n0_values:
                        if self._stop_requested:
                            return

                        # --- 核心修复：使用 math.isclose 进行模糊比较 ---
                        target_key = (cube_size, fractal_dim, n0)
                        completed_count = 0
                        for task_key in self.completed_tasks:
                            if (len(task_key) == 3 and
                                math.isclose(task_key[0], target_key[0], rel_tol=1e-9) and
                                math.isclose(task_key[1], target_key[1], rel_tol=1e-9) and
                                math.isclose(task_key[2], target_key[2], rel_tol=1e-9)):
                                completed_count += 1
                        # --- 修复结束 ---
                        
                        remaining_repeats = self.repeat_count - completed_count
                        if remaining_repeats <= 0:
                            # 注意：这里需要加上已完成的次数，以保证进度条总数正确
                            current_simulation_count += self.repeat_count - remaining_repeats 
                            continue

                        task_params = {
                            'cube_size': cube_size, 'fractal_dim': fractal_dim, 'n0': n0,
                            'repeat_count': remaining_repeats,
                            'gen_iterations': self.gen_iterations, 
                            'calc_iterations': self.calc_iterations,
                            'num_slices': self.num_slices, # <--- 传递切片数

                            
                            # (保留)
                            'random_aspect_ratio': self.random_aspect_ratio,
                            'aspect_ratio_base': self.aspect_ratio_base,
                            'aspect_ratio_variation': self.aspect_ratio_variation,

                            # (新增)
                            'is_isotropic': self.is_isotropic,
                            'mean_inclination': self.mean_inclination,
                            'mean_azimuth': self.mean_azimuth,
                            'kappa': self.kappa,
                            'use_advanced_model': self.use_advanced_model, # <<< 新增

                            'base_seed': int(time.time() * 1000) + task_id_counter,
                            'task_id': task_id_counter,
                            'completed_count': completed_count
                        }
                        task_id_counter += 1

                        results_list = _run_simulation_task(task_params)

                        for result in results_list:
                            if self._stop_requested:
                                return
                            
                            self.results.append(result)
                            
                            if backup_manager:
                                backup_manager.append_result(result)
                            
                            self.single_result_ready.emit(result)
                            
                            current_simulation_count += 1
                            self.calculation_progress.emit(current_simulation_count, total_simulations, "")

            if not self._stop_requested:
                self.calculation_finished.emit(len(self.results))
                
        except Exception as e:
            import traceback
            error_msg = f"单线程计算过程中发生错误: {str(e)}\n{traceback.format_exc()}"
            self.calculation_failed.emit(error_msg)
        finally:
            pass
    



class Study2D3DParallelCalculationThread(QThread):
    """2D-3D关系研究多进程并行计算主线程"""
    calculation_progress = pyqtSignal(int, int, str)  # 进度信号：当前进度，总进度，状态文本
    single_result_ready = pyqtSignal(dict)  # 单个结果完成信号
    calculation_finished = pyqtSignal(int)  # 所有计算完成信号，传递结果数量
    calculation_failed = pyqtSignal(str)  # 计算失败信号
    
# 文件: FD-Advanced_3D.py (Study2D3DParallelCalculationThread 类)

    def __init__(self, cube_lengths, fractal_dims, n0_values, repeat_count, 
                 gen_iterations, calc_iterations, num_slices,
                 calc_slice_profile, 
                 calc_center_only, # <--- 必须添加这个参数 (第9个位置)
                 random_aspect_ratio, aspect_ratio_base, aspect_ratio_variation,
                 is_isotropic, mean_inclination, mean_azimuth, kappa,
                 use_advanced_model, # <<< 新增
                 results_queue,      # <--- 必须在第17个位置
                 max_processes=4, backup_file_path=None, gui_input_params=None, completed_tasks=None):
        super().__init__()
        self.cube_lengths = cube_lengths
        self.fractal_dims = fractal_dims
        self.n0_values = n0_values
        self.repeat_count = repeat_count
        self.gen_iterations = gen_iterations
        self.calc_iterations = calc_iterations
        self.num_slices = num_slices
        self.calc_slice_profile = calc_slice_profile
        self.calc_center_only = calc_center_only # 保存新参数
        
        self.random_aspect_ratio = random_aspect_ratio
        self.aspect_ratio_base = aspect_ratio_base
        self.aspect_ratio_variation = aspect_ratio_variation
        
        self.is_isotropic = is_isotropic
        self.mean_inclination = mean_inclination
        self.mean_azimuth = mean_azimuth
        self.kappa = kappa
        self.use_advanced_model = use_advanced_model
        
        self.results_queue = results_queue
        self.max_processes = max_processes
        self.results = []
        self._stop_requested = False
        self.backup_file_path = backup_file_path
        self.completed_tasks = completed_tasks or []
        
        self.calculation_params = {
            "cube_lengths": cube_lengths, "fractal_dims": fractal_dims,
            "n0_values": n0_values, "repeat_count": repeat_count,
            "gen_iterations": gen_iterations, "calc_iterations": calc_iterations,
            "num_slices": num_slices,
            "calc_slice_profile": calc_slice_profile,
            "calc_center_only": calc_center_only, # 存入参数字典
            "random_aspect_ratio": random_aspect_ratio,
            "aspect_ratio_base": aspect_ratio_base,
            "aspect_ratio_variation": aspect_ratio_variation,
            "is_isotropic": is_isotropic,
            "mean_inclination": mean_inclination,
            "mean_azimuth": mean_azimuth,
            "kappa": kappa,
            "use_advanced_model": use_advanced_model, # <<< 新增
            "max_processes": max_processes,
            "gui_input_params": gui_input_params or {}
        }
        
        self.task_queue = []
        self.total_simulations = 0
        self._generate_task_queue()
    
    def _generate_task_queue(self):
        """生成任务队列（移除不必要的参数传递）"""
        task_id = 0
        import time
        master_seed = int(time.time() * 1000)
        rng_manager = RandomStateManager(master_seed)
        
        for cube_size in self.cube_lengths:
            for fractal_dim in self.fractal_dims:
                for n0 in self.n0_values:
                    # 统计该参数组合已完成的重复次数
                    # --- 核心修复：使用 math.isclose 进行模糊比较 ---
                    target_key = (cube_size, fractal_dim, n0)
                    completed_count = 0
                    for task_key in self.completed_tasks:
                        # 逐个元素比较，因为浮点数可能存在微小差异
                        if (len(task_key) == 3 and
                            math.isclose(task_key[0], target_key[0], rel_tol=1e-9) and
                            math.isclose(task_key[1], target_key[1], rel_tol=1e-9) and
                            math.isclose(task_key[2], target_key[2], rel_tol=1e-9)):
                            completed_count += 1
                    # --- 修复结束 ---
                    
                    if completed_count >= self.repeat_count:
                        continue
                    
                    remaining_repeats = self.repeat_count - completed_count
                    task_seed = rng_manager.fracture_rng.integers(0, 2**32 - 1)
                    
                    task = {
                        'cube_size': cube_size, 'fractal_dim': fractal_dim, 'n0': n0,
                        'repeat_count': remaining_repeats,
                        'gen_iterations': self.gen_iterations, 
                        'calc_iterations': self.calc_iterations,
                        'num_slices': self.num_slices, # <--- 传递切片数
                        'calc_slice_profile': self.calc_slice_profile, # <--- 传递计算切片剖面参数
                        'calc_center_only': self.calc_center_only, # <--- 传递中心域优化参数
                        
                        # (保留)
                        'random_aspect_ratio': self.random_aspect_ratio,
                        'aspect_ratio_base': self.aspect_ratio_base,
                        'aspect_ratio_variation': self.aspect_ratio_variation,
                        
                        # (新增)
                        'is_isotropic': self.is_isotropic,
                        'mean_inclination': self.mean_inclination,
                        'mean_azimuth': self.mean_azimuth,
                        'kappa': self.kappa,
                        'use_advanced_model': self.use_advanced_model, # <<< 新增

                        'base_seed': task_seed,
                        'task_id': task_id,
                        'backup_file_path': self.backup_file_path,
                        'completed_count': completed_count
                    }
                    self.task_queue.append(task)
                    task_id += 1
        
        self.total_simulations = sum(task['repeat_count'] for task in self.task_queue)
    
    def stop_calculation(self):
        """请求优雅地停止计算"""
        print("收到停止计算请求...")
        self._stop_requested = True # 这个UI线程标志仍然有用
        
        # <<< 核心修改: 不再调用 terminate()，而是设置共享事件 >>>
        if hasattr(self, 'stop_event') and self.stop_event:
            print("正在设置停止事件，通知所有工作进程...")
            self.stop_event.set() # 这就是“升起停止信号旗”的操作
        # <<< 修改结束 >>>

        # --- VVVV 在这里添加以下新代码 VVVV ---
        if hasattr(self, 'listener'):
            self.listener.stop()
        # --- ^^^^ 完成添加 ^^^^ ---

        # <<< 核心修改: 彻底移除 terminate() 和 join() 的调用 >>>
        # if hasattr(self, '_current_pool') and self._current_pool is not None:
        #     print("正在终止进程池...")
        #     self._current_pool.terminate()
        #     self._current_pool.join()
    
# 文件: FD-Advanced_3D.py
# 请用此最终修复版完整替换 Study2D3DParallelCalculationThread.run 函数

# --- START OF REPLACEMENT CODE FOR run() method ---
    def run(self):
        """运行多进程并行计算 (最终修复版：只管派发任务)"""
        pool = None
        try:
            if not self.task_queue:
                print("任务队列为空，无需执行计算。")
                self.calculation_finished.emit(0) # 告知主线程已完成（完成了0个新任务）
                return

            if self.backup_file_path:
                try:
                    _ = SQLiteBackupManager(self.backup_file_path, self.calculation_params)
                except Exception as e:
                    import traceback
                    error_msg = f"无法写入计算参数到备份文件: {str(e)}\n{traceback.format_exc()}"
                    self.calculation_failed.emit(error_msg)
                    return

            self.stop_event = multiprocessing.Event()
            actual_processes = min(self.max_processes, os.cpu_count() or 1)
            
            # 初始化进程池，将全局队列和停止事件传递给子进程
            pool = Pool(processes=actual_processes, 
                        initializer=init_worker, 
                        initargs=(self.results_queue, self.stop_event))
            self._current_pool = pool
            
            # 遍历任务队列，将每个任务字典单独提交给进程池
            for task in self.task_queue:
                if self._stop_requested:
                    break
                pool.apply_async(multiprocess_single_calculation_ipc, (task,))

            pool.close()  # 关闭池，表示不再接受新任务
            pool.join()   # 阻塞，直到所有子进程完成它们被分配的任务

            if not self._stop_requested:
                # 告知主线程，所有任务都已成功派发并完成
                # 传递已提交的任务总数
                self.calculation_finished.emit(self.total_simulations)
            else:
                # 如果是用户中止，也发完成信号，让主GUI可以清理状态
                print("计算已由用户中止。")
                self.calculation_finished.emit(-1) # 用-1或其他特殊值表示中止
                
        except Exception as e:
            import traceback
            error_msg = f"多进程管理线程发生严重错误: {str(e)}\n{traceback.format_exc()}"
            self.calculation_failed.emit(error_msg)
        finally:
            if pool:
                # 紧急情况下终止仍在运行的子进程
                pool.terminate()
                pool.join()
            self._current_pool = None
            print("多进程管理线程已完成任务并退出。")
    # --- END OF REPLACEMENT CODE ---





# ==================== 原有类定义 ====================

class MplCanvas(FigureCanvas):
    """matplotlib画布"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100, is_3d=True):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if is_3d:
            self.axes = self.fig.add_subplot(111, projection='3d')
        else:
            self.axes = self.fig.add_subplot(111)
        
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


class FractalFractureGeneratorGUI(QMainWindow):
    """基于分形维数理论的裂缝生成器GUI"""
    
    def __init__(self):
        super().__init__()

        import time
        master_seed = int(time.time())
        print(f"应用程序启动，创建主随机管理器，种子: {master_seed}")
        self.rng_manager = RandomStateManager(master_seed)

        self.generator = FractalBasedFractureGenerator()
        self.fractal_calculator = FractalDimension3DCalculator()
        self.fractal_2d_calculator = FractalDimension2DCalculator(iterations=4)  # 使用新的构造函数，iterations参数现在控制计算层级
        self.fracture_exporter = FractureExporter()  # 添加导出器
        self.drilling_analyzer = DrillingAnalyzer(rng_manager=self.rng_manager)
        self.canvases = {}  # 动态创建的画布
        self.tab_widget = None  # 标签页控件
        self.calculation_thread = None  # 计算线程
        self.generation_thread = None  # 生成线程
        self.study_2d3d_thread = None  # 2D-3D关系研究线程
        self.latest_fractures = []  # 最新生成的裂缝
        self.latest_detailed_stats = []  # 最新的详细统计

        self.unfolding_analysis_btn = None

        # <<< 在这里添加下面的代码 >>>
        self.latest_calculated_dim = None
        self.latest_fit_data = None
        self.latest_levels = None
        self.latest_face_analysis_data = None

        # <<< 添加结束 >>>
        
        # 六个面投影分析相关变量 - 只保留分组平均画布
        self.group_fractal_canvases = {}  # YOZ、XOZ、XOY三组的分形维数画布
        self.group_data_canvases = {}  # YOZ、XOZ、XOY三组的数据显示画布
        
        # 统计分析相关变量
        self.inclination_canvas = None  # 倾角分布画布
        self.azimuth_canvas = None  # 方位角分布画布
        self.fractal_3d_canvas = None  # 三维分形维数拟合画布
        
        # 钻孔分析相关变量
        self.drilling_probability_canvas = None  # 钻孔概率画布
        self.drilling_table = None  # 钻孔概率表格
        self.drilling_analysis_data = None  # 钻孔分析数据
        
        # 2D-3D关系研究结果显示相关变量
        self.study_results = []  # 原始结果数据
        self.study_aggregated_results = []  # 聚合统计结果
        self.completed_tasks_offset = 0  # <--- 新增这一行
        self.current_session_timestamp = None  # <<< 新增：用于存储当前会话的时间戳
        

        self.current_display_mode = "统计汇总(平均值±标准差)"  # 默认显示模式
        self.current_backup_file = None  # 当前备份文件路径
        
        # 截面结果窗口引用
        self.section_result_dialog = None
        self.enhanced_section_result_dialog = None  # 增强版截面结果窗口引用
        
        # 滑动条相关变量
        self.auto_capture_enabled = False
        self.slider_updating = False  # 防止滑动条和输入框之间的循环更新
        
        # 为钻孔分析面板和控制按钮添加属性
        self.drilling_study_btn = None
        # --- 新增：初始化所有可能在钻孔标签页中创建的UI组件 ---
        self.generate_random_point_btn = None
        self.start_analysis_btn = None
        self.clear_analysis_btn = None
        self.show_non_intersected_check = None
        self.fracture_opacity_spin = None
        self.non_intersected_opacity_spin = None
        self.random_point_label = None
        self.analysis_result_label = None
        # --- 初始化结束 ---
        # --- 新增：初始化计时器 ---
        from PyQt5.QtCore import QTimer
        self.runtime_timer = QTimer(self)
        self.runtime_timer.setInterval(100) # 每100毫秒更新一次
        self.runtime_timer.timeout.connect(self.update_timer_display)
        
        
        self.result_buffer = deque()  # 1. 创建线程安全的缓冲区实例
        
        self.ui_update_timer = QTimer(self) # 2. 创建UI刷新定时器
        self.ui_update_timer.setInterval(100) # 3. 设置刷新频率 (100ms = 每秒10次)
        # 4. 将定时器的超时信号连接到一个新的批量处理函数上
        self.ui_update_timer.timeout.connect(self._process_result_buffer_and_update_ui) 
        # --- 新增结束 ---

        self.init_ui()


        
        # --- 新增代码 ---
        # 将 TranslationManager 的创建和关联移到这里
        self.tr_manager = TranslationManager(QApplication.instance())
        self.tr_manager.language_changed.connect(self.retranslate_ui)
        # --- 结束新增 ---

        # --- 关键修复 ---
        # 在窗口显示前，手动调用一次 retranslate_ui 来设置所有控件的初始文本
        self.retranslate_ui()
        # --- 修复结束 ---
        # --- 核心修复：将多进程通信设施提升到主窗口级别 ---
        # 使用 Manager 来创建队列，这在复杂应用中更健壮
        self.mp_manager = multiprocessing.Manager()
        self.results_queue = self.mp_manager.Queue()
        
        # 创建一个常驻的队列监听器和它的线程
        self.queue_listener = QueueListener(self.results_queue, self.result_buffer)
        self.listener_thread = QThread()
        self.queue_listener.moveToThread(self.listener_thread)
        
        # 连接信号：监听器收到新结果后，由主GUI线程的槽函数处理
        
        # 启动监听线程，它将在整个应用程序生命周期内运行
        self.listener_thread.started.connect(self.queue_listener.run)
        self.listener_thread.start()
        print("全局多进程队列监听器已启动。")

    def update_timer_display(self):
        """由QTimer触发，用于实时更新耗时显示"""
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            # 直接显示时间，不带文字
            self.timer_label.setText(f"{elapsed_time:.1f} s")

    def stop_all_processes(self):
        """统一的停止函数，中断所有当前活动并恢复UI"""
        was_stopped = False

        # --- 核心修改：停止QTimer ---
        if self.runtime_timer.isActive():
            self.runtime_timer.stop()
        # --- 修改结束 ---

        # 尝试停止生成线程
        if self.generation_thread and self.generation_thread.isRunning():
            print("正在停止生成线程...")
            self.generation_thread.stop() # 请求协作式停止
            self.generation_thread.wait(1000) # 等待1秒
            if self.generation_thread and self.generation_thread.isRunning():
                print("协作式停止失败，强制终止生成线程...")
                self.generation_thread.terminate() # 强制终止
                self.generation_thread.wait()
            was_stopped = True
        self.generation_thread = None

        # 尝试停止计算线程
        if self.calculation_thread and self.calculation_thread.isRunning():
            print("正在停止计算线程...")
            self.calculation_thread.terminate() # 计算线程可以直接强制终止
            self.calculation_thread.wait()
            was_stopped = True
        self.calculation_thread = None

        if was_stopped:
            self.stats_text.setText(self.tr("操作已由用户中止。"))
            # 如果是用户中止，也显示当时的时间
            if self.start_time:
                elapsed_time = time.time() - self.start_time
                self.timer_label.setText(f"{elapsed_time:.1f} s")
                self.timer_label.setVisible(True)
            
        # 统一恢复UI状态
        self.progress_bar.setVisible(False)
        if hasattr(self, 'stop_btn'):
            self.stop_btn.setEnabled(False)
        if hasattr(self, 'generate_btn'):
            self.generate_btn.setEnabled(True)

    def change_language(self, lang_code):
        """槽函数，用于响应语言切换请求"""
        if hasattr(self, 'tr_manager'):
            self.tr_manager.load_translation(lang_code)

    # 我们将创建一个新的方法来更新UI文本
    def retranslate_ui(self):
        """
        这个方法将在语言切换后被调用，
        用于更新所有界面元素的文本。
        我们将在第二阶段填充它。
        """
        print("Retranslating UI...")
        # 这是一个占位符，之后会添加实际的更新代码
        self.setWindowTitle(self.tr('基于分形维数理论的空间椭圆裂缝生成器 - 增强版（含钻孔分析）'))
        # ... 其他所有UI元素的文本更新 ...
    
    @staticmethod
    def _save_table_to_csv(parent, table_widget, default_filename_prefix):
        """
        REFACTORED: 通用函数，用于将QTableWidget的内容保存为CSV。
        """
        # --- 核心修改 ---
        dialog_title = QCoreApplication.translate("FractalFractureGeneratorGUI", "保存表格 - {0}").format(default_filename_prefix)
        file_filter = QCoreApplication.translate("FractalFractureGeneratorGUI", "CSV文件 (*.csv)")
        filename, _ = QFileDialog.getSaveFileName(parent, dialog_title, f"{default_filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", file_filter)
        
        if not filename:
            return

        try:
            headers = [table_widget.horizontalHeaderItem(col).text() for col in range(table_widget.columnCount())]
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for row in range(table_widget.rowCount()):
                    row_data = [table_widget.item(row, col).text() if table_widget.item(row, col) else "" for col in range(table_widget.columnCount())]
                    writer.writerow(row_data)
            
            success_message = QCoreApplication.translate("FractalFractureGeneratorGUI", "表格已保存到:\n{0}").format(filename)
            QMessageBox.information(parent, QCoreApplication.translate("FractalFractureGeneratorGUI", "保存成功"), success_message)
        except Exception as e:
            fail_message = QCoreApplication.translate("FractalFractureGeneratorGUI", "保存失败:\n{0}").format(str(e))
            QMessageBox.critical(parent, QCoreApplication.translate("FractalFractureGeneratorGUI", "保存失败"), fail_message)

    # 替换 _save_figure_to_file 函数
    @staticmethod
    def _save_figure_to_file(parent, figure, default_filename_prefix):
        """
        REFACTORED: 通用函数，用于将Matplotlib Figure对象保存为图片。
        """
        # --- 核心修改 ---
        dialog_title = QCoreApplication.translate("FractalFractureGeneratorGUI", "保存图表")
        file_filter = QCoreApplication.translate("FractalFractureGeneratorGUI", "PNG图片 (*.png);;PDF文件 (*.pdf)")
        filename, _ = QFileDialog.getSaveFileName(parent, dialog_title, f"{default_filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", file_filter)
        
        if not filename:
            return

        try:
            figure.savefig(filename, dpi=300, bbox_inches='tight')
            success_message = QCoreApplication.translate("FractalFractureGeneratorGUI", "图表已保存到:\n{0}").format(filename)
            QMessageBox.information(parent, QCoreApplication.translate("FractalFractureGeneratorGUI", "保存成功"), success_message)
        except Exception as e:
            fail_message = QCoreApplication.translate("FractalFractureGeneratorGUI", "保存失败:\n{0}").format(str(e))
            QMessageBox.critical(parent, QCoreApplication.translate("FractalFractureGeneratorGUI", "保存失败"), fail_message)

    def init_ui(self):
        """初始化用户界面"""

        # 创建菜单栏
        menu_bar = self.menuBar()
        # 修改: 创建时不设置文本，而是保存实例变量
        self.settings_menu = menu_bar.addMenu('') 
        self.language_menu = self.settings_menu.addMenu('')

        self.action_zh = self.language_menu.addAction('')
        self.action_zh.triggered.connect(lambda: self.change_language('zh'))

        self.action_en = self.language_menu.addAction('')
        self.action_en.triggered.connect(lambda: self.change_language('en'))
        # --- 结束修改 ---

        self.setWindowTitle('基于分形维数理论的空间椭圆裂缝生成器 - 增强版（含钻孔分析）')
        self.setGeometry(50, 50, 2400, 1600)  # 调整窗口尺寸以适应两栏布局

        # 创建主widget和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 修改后：创建水平分割器（两栏布局）
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
        main_widget.setLayout(main_layout)

        # 左侧参数面板
        self.create_parameter_panel(main_splitter)

        # 中间显示区域（现在也是右侧）
        self.create_display_area(main_splitter)

        # 修改后：不再创建右侧的钻孔分析面板
        # self.create_drilling_analysis_panel(main_splitter)

        # 修改后：设置分割器比例为参数面板:显示区域 = 450:1950
        main_splitter.setSizes([600, 1950])

    def create_parameter_panel(self, parent):
        """创建参数输入面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(600)
        
        layout = QVBoxLayout()
        
        # --- 分形参数组 ---
        self.fractal_group = QGroupBox() # 修改1: 移除硬编码文本
        fractal_layout = QFormLayout()
        
        self.cube_size_spin = QDoubleSpinBox()
        self.cube_size_spin.setRange(0.000001, 1000000000000000000000000000000.0)
        self.cube_size_spin.setValue(1)
        self.cube_size_spin.setDecimals(6)
        self.cube_size_label = QLabel() # 修改2: 创建 QLabel 实例
        fractal_layout.addRow(self.cube_size_label, self.cube_size_spin) # 修改3: 添加实例化的label
        
        self.fractal_dim_spin = QDoubleSpinBox()
        self.fractal_dim_spin.setRange(0.01, 2.99)
        self.fractal_dim_spin.setValue(2.7)
        self.fractal_dim_spin.setDecimals(3)
        self.fractal_dim_spin.setSingleStep(0.1)
        self.fractal_dim_label = QLabel()
        fractal_layout.addRow(self.fractal_dim_label, self.fractal_dim_spin)
        
        self.n0_spin = QDoubleSpinBox()
        self.n0_spin.setRange(0.1, 1000000000000000000000000000000.0)
        self.n0_spin.setValue(1)
        self.n0_spin.setDecimals(3)
        self.n0_spin.setSingleStep(0.1)
        self.n0_label = QLabel()
        fractal_layout.addRow(self.n0_label, self.n0_spin)
        
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 5)
        self.iterations_spin.setValue(3)
        self.iterations_label = QLabel()
        fractal_layout.addRow(self.iterations_label, self.iterations_spin)
        
        self.calc_iterations_spin = QSpinBox()
        self.calc_iterations_spin.setRange(1, 8)
        self.calc_iterations_spin.setValue(3)
        self.calc_iterations_label = QLabel()
        fractal_layout.addRow(self.calc_iterations_label, self.calc_iterations_spin)
        
        self.iterations_spin.valueChanged.connect(lambda v: self._sync_spinbox_value('calc_iterations_spin', v))
        self.calc_iterations_spin.valueChanged.connect(lambda v: self._sync_spinbox_value('iterations_spin', v))

        self.main_num_slices_spin = QSpinBox()
        self.main_num_slices_spin.setRange(3, 201)
        self.main_num_slices_spin.setValue(101)
        self.main_num_slices_spin.setSingleStep(2) # 建议奇数
        self.main_num_slices_label = QLabel()
        fractal_layout.addRow(self.main_num_slices_label, self.main_num_slices_spin)
        
        self.fractal_group.setLayout(fractal_layout)
        layout.addWidget(self.fractal_group)
        
        # --- 椭圆形态参数组 ---
# 文件: FD-Advanced_3D.py (在 create_parameter_panel 方法中)

        # --- 椭圆形态参数组 (已重构) ---
        self.ellipse_group = QGroupBox()
        ellipse_layout = QVBoxLayout()
        
        # 1. (保留) 长短轴比例部分
        aspect_ratio_layout = QVBoxLayout()
        self.random_aspect_ratio_check = QCheckBox()
        self.random_aspect_ratio_check.setChecked(False)
        self.random_aspect_ratio_check.toggled.connect(self.toggle_aspect_ratio_controls)
        aspect_ratio_layout.addWidget(self.random_aspect_ratio_check)
        
        self.aspect_ratio_controls = QWidget()
        aspect_ratio_form = QFormLayout()
        
        self.aspect_ratio_base_spin = QDoubleSpinBox()
        self.aspect_ratio_base_spin.setRange(1.0, 10.0)
        self.aspect_ratio_base_spin.setValue(1.0)
        self.aspect_ratio_base_spin.setDecimals(2)
        self.aspect_ratio_base_spin.setSingleStep(0.1)
        self.aspect_ratio_base_label = QLabel()
        aspect_ratio_form.addRow(self.aspect_ratio_base_label, self.aspect_ratio_base_spin)
        
        self.aspect_ratio_variation_spin = QDoubleSpinBox()
        self.aspect_ratio_variation_spin.setRange(0.0, 5.0)
        self.aspect_ratio_variation_spin.setValue(0.0)
        self.aspect_ratio_variation_spin.setDecimals(2)
        self.aspect_ratio_variation_spin.setSingleStep(0.1)
        self.aspect_ratio_variation_label = QLabel()
        aspect_ratio_form.addRow(self.aspect_ratio_variation_label, self.aspect_ratio_variation_spin)
        
        self.aspect_ratio_controls.setLayout(aspect_ratio_form)
        self.aspect_ratio_controls.setVisible(True) # 默认可见
        aspect_ratio_layout.addWidget(self.aspect_ratio_controls)
        ellipse_layout.addLayout(aspect_ratio_layout)

        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        ellipse_layout.addWidget(separator)
        
        # 2. (新增) 产状分布 (Fisher 分布) 部分
        self.orientation_group = QGroupBox() # 新的分组框
        orientation_layout = QVBoxLayout()

        # 复选框行布局
        checkbox_layout = QHBoxLayout()
        
        self.isotropic_check = QCheckBox()
        self.isotropic_check.setChecked(True) # 默认勾选全随机
        self.isotropic_check.toggled.connect(self.toggle_orientation_mode)
        checkbox_layout.addWidget(self.isotropic_check)
        
        self.advanced_model_check = QCheckBox()
        self.advanced_model_check.setChecked(False)
        self.advanced_model_check.toggled.connect(self.toggle_advanced_mode)
        checkbox_layout.addWidget(self.advanced_model_check)
        
        orientation_layout.addLayout(checkbox_layout)

        self.anisotropic_controls = QWidget() # 包裹各向异性参数的容器
        anisotropic_form = QFormLayout()

        self.mean_inclination_spin = QDoubleSpinBox()
        self.mean_inclination_spin.setRange(0.0, 180.0)
        self.mean_inclination_spin.setValue(90.0)
        self.mean_inclination_spin.setDecimals(1)
        self.mean_inclination_label = QLabel()
        anisotropic_form.addRow(self.mean_inclination_label, self.mean_inclination_spin)

        self.mean_azimuth_spin = QDoubleSpinBox()
        self.mean_azimuth_spin.setRange(0.0, 360.0)
        self.mean_azimuth_spin.setValue(0.0)
        self.mean_azimuth_spin.setDecimals(1)
        self.mean_azimuth_label = QLabel()
        anisotropic_form.addRow(self.mean_azimuth_label, self.mean_azimuth_spin)

        self.kappa_spin = QDoubleSpinBox()
        self.kappa_spin.setRange(0, 10000.0)
        self.kappa_spin.setValue(20.0)
        self.kappa_spin.setDecimals(1)
        self.kappa_label = QLabel()
        anisotropic_form.addRow(self.kappa_label, self.kappa_spin)
        
        self.anisotropic_controls.setLayout(anisotropic_form)
        orientation_layout.addWidget(self.anisotropic_controls)
        
        self.orientation_group.setLayout(orientation_layout)
        ellipse_layout.addWidget(self.orientation_group)
        
        self.ellipse_group.setLayout(ellipse_layout)
        layout.addWidget(self.ellipse_group)

        # 初始状态下禁用各向异性控件
        self.anisotropic_controls.setEnabled(False)
        
        # --- 计算选项组 ---
        self.calc_options_group = QGroupBox()
        calc_options_layout = QVBoxLayout()

        self.calc_options_group.setLayout(calc_options_layout)
        layout.addWidget(self.calc_options_group)
        
        # --- 截面捕获组 ---
        self.section_capture_group = QGroupBox()
        section_capture_layout = QFormLayout()
        
        self.section_type_combo = QComboBox()
        self.section_type_combo.currentIndexChanged.connect(self.on_section_type_changed)
        self.section_type_label = QLabel()
        section_capture_layout.addRow(self.section_type_label, self.section_type_combo)
        
        self.coordinate_spin = QDoubleSpinBox()
        self.coordinate_spin.setRange(0, 1000)
        self.coordinate_spin.setValue(0.0)
        self.coordinate_spin.setDecimals(2)
        self.coordinate_spin.setSingleStep(0.1)
        self.coordinate_spin.valueChanged.connect(self.on_coordinate_changed)
        self.coordinate_label = QLabel()
        section_capture_layout.addRow(self.coordinate_label, self.coordinate_spin)
        
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.setValue(0)
        self.position_slider.setTickPosition(QSlider.TicksBelow)
        self.position_slider.setTickInterval(100)
        self.position_slider.setEnabled(False)
        self.position_slider.valueChanged.connect(self.on_slider_changed)
        self.position_slider_label = QLabel()
        section_capture_layout.addRow(self.position_slider_label, self.position_slider)
        
        self.auto_capture_checkbox = QCheckBox()
        self.auto_capture_checkbox.setChecked(False)
        self.auto_capture_checkbox.setEnabled(False)
        self.auto_capture_checkbox.stateChanged.connect(self.on_auto_capture_changed)
        section_capture_layout.addRow("", self.auto_capture_checkbox) 
        
        self.auto_calc_fractal_checkbox = QCheckBox()
        self.auto_calc_fractal_checkbox.setChecked(False)
        self.auto_calc_fractal_checkbox.setEnabled(False)
        section_capture_layout.addRow("", self.auto_calc_fractal_checkbox)
        
        self.capture_section_btn = QPushButton()
        self.capture_section_btn.clicked.connect(self.capture_section)
        self.capture_section_btn.setEnabled(False)
        section_capture_layout.addRow("", self.capture_section_btn)

                # --- 新增：切片趋势分析按钮 ---
        self.analyze_trend_btn = QPushButton()
        self.analyze_trend_btn.clicked.connect(self.plot_slice_trend)
        self.analyze_trend_btn.setEnabled(False) # 初始禁用
        section_capture_layout.addRow("", self.analyze_trend_btn)
        # ---------------------------
        
        self.section_capture_group.setLayout(section_capture_layout)
        layout.addWidget(self.section_capture_group)
        
        # --- 导出选项组 ---
        self.export_options_group = QGroupBox()
        export_options_layout = QVBoxLayout()
        
        export_format_layout = QHBoxLayout()
        self.export_format_label = QLabel()
        export_format_layout.addWidget(self.export_format_label)
        self.export_format_combo = QComboBox()
        export_format_layout.addWidget(self.export_format_combo)
        export_options_layout.addLayout(export_format_layout)
        
        self.export_btn = QPushButton()
        self.export_btn.clicked.connect(self.export_model)
        self.export_btn.setEnabled(False)
        export_options_layout.addWidget(self.export_btn)
        
        self.export_options_group.setLayout(export_options_layout)
        layout.addWidget(self.export_options_group)

        # --- 控制按钮 ---
        # 使用网格布局，更灵活
        button_layout = QGridLayout()
        button_layout.setSpacing(8)
        button_layout.setContentsMargins(8, 8, 8, 8)

        # 主操作按钮（合并了生成和分析）
        self.generate_btn = QPushButton()
        self.generate_btn.clicked.connect(self.generate_fractures)
        button_layout.addWidget(self.generate_btn, 0, 0) # 第0行，第0列

        # 统一的停止按钮
        self.stop_btn = QPushButton()
        self.stop_btn.clicked.connect(self.stop_all_processes) # <--- 连接到新的统一停止函数
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn, 0, 1) # 第0行，第1列

        # 清除按钮
        self.clear_btn = QPushButton()
        self.clear_btn.clicked.connect(self.clear_display)
        button_layout.addWidget(self.clear_btn, 1, 0) # 第1行，第0列

        # 捕获三维视图按钮
        self.capture_3d_view_btn = QPushButton()
        self.capture_3d_view_btn.clicked.connect(self.capture_current_3d_view)
        self.capture_3d_view_btn.setEnabled(False) # 初始禁用
        button_layout.addWidget(self.capture_3d_view_btn, 1, 1) # 第1行，第1列

        # 2D-3D关系研究按钮
        self.relation_2d3d_btn = QPushButton()
        self.relation_2d3d_btn.clicked.connect(self.open_2d3d_study)
        button_layout.addWidget(self.relation_2d3d_btn, 2, 0)

        # 钻孔模拟研究按钮
        self.drilling_study_btn = QPushButton()

        self.unfolding_analysis_btn = QPushButton()
        self.unfolding_analysis_btn.clicked.connect(self.open_unfolding_analyzer)
        self.unfolding_analysis_btn.setEnabled(False) # 初始禁用
        button_layout.addWidget(self.unfolding_analysis_btn, 3, 0, 1, 2) # 占据第3行，跨越两列

        self.drilling_study_btn.clicked.connect(self.open_drilling_study_tab)
        self.drilling_study_btn.setEnabled(False)
        button_layout.addWidget(self.drilling_study_btn, 2, 1)
        

        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # --- 新增：计时器显示 ---
        self.timer_label = QLabel("耗时: 0.0 s")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-weight: bold; color: #2E86AB; margin-top: 5px;")
        self.timer_label.setVisible(False) # 默认隐藏
        layout.addWidget(self.timer_label)
        
        # --- 统计信息显示 ---
        self.stats_group = QGroupBox()
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Courier New", 9))
        self.stats_text.setMinimumHeight(200)
        stats_layout.addWidget(self.stats_text)
        
        self.stats_group.setLayout(stats_layout)
        layout.addWidget(self.stats_group)
        
        layout.setStretchFactor(self.stats_group, 1)
        
        layout.addStretch(0)
        panel.setLayout(layout)
        parent.addWidget(panel)
# 修正后的代码
    def open_2d3d_study(self):
        """按需创建或切换到2D-3D关系研究标签页"""
        tab_name = self.tr("2D-3D关系研究")
        tab_found = False

        # 1. 遍历所有已存在的标签页
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == tab_name:
                self.tab_widget.setCurrentIndex(i)
                tab_found = True
                break

        # 2. 如果循环完成仍未找到，说明标签页不存在，此时才调用创建方法
        if not tab_found:
            self.create_2d3d_study_tab()
            
            # ========== FIX START: 添加缺失的调用 ==========
            # 在新标签页创建完成后，调用主翻译函数来填充其所有控件的文本。
            self.retranslate_ui()
            # ========== FIX END ==========```
    
    # THIS IS THE NEW CODE
    def create_2d3d_study_tab(self):
        """创建2D-3D关系研究标签页"""
        study_widget = QWidget()
        
        # --- 核心修改: 使用 objectName 代替成员变量 ---
        study_widget.setObjectName("study_2d3d_tab_widget")
        
        main_layout = QHBoxLayout()
        params_panel = self.create_2d3d_params_panel(study_widget)
        main_layout.addWidget(params_panel)
        results_area = self.create_2d3d_results_area(study_widget)
        main_layout.addWidget(results_area)
        main_layout.setStretch(0, 3)
        main_layout.setStretch(1, 7)
        study_widget.setLayout(main_layout)
        
        # 使用临时标题
        self.tab_widget.addTab(study_widget, "...")
        
        self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)
    
    def create_2d3d_params_panel(self, parent):
        """创建2D-3D关系研究参数设置面板"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(500)
        
        layout = QVBoxLayout()
        
        # 标题
        self.study_title_label = QLabel() # 修改: 移除文本
        self.study_title_label.setAlignment(Qt.AlignCenter)
        self.study_title_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #2E86AB; margin: 10px;")
        layout.addWidget(self.study_title_label)
        
        # 参数输入组
        self.study_params_group = QGroupBox() # 修改: 移除文本
        self.study_params_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        params_layout = QFormLayout()
        params_layout.setSpacing(5)
        params_layout.setContentsMargins(8, 8, 8, 8)
        
        # 立方体长度输入
        self.study_cube_lengths_input = QLineEdit()
        self.study_cube_lengths_input.setText("1")
        self.study_cube_lengths_input.setStyleSheet("font-size: 14px;")
        self.study_cube_lengths_label = QLabel() # 修改: 创建实例
        params_layout.addRow(self.study_cube_lengths_label, self.study_cube_lengths_input)
        
        # 分形维数输入
        self.study_fractal_dims_input = QLineEdit()
        self.study_fractal_dims_input.setText("(2.1,2.9,0.1)")
        self.study_fractal_dims_input.setStyleSheet("font-size: 14px;")
        self.study_fractal_dims_label = QLabel() # 修改: 创建实例
        params_layout.addRow(self.study_fractal_dims_label, self.study_fractal_dims_input)
        
        # 分形初值输入
        self.study_n0_values_input = QLineEdit()
        self.study_n0_values_input.setText("1")
        self.study_n0_values_input.setStyleSheet("font-size: 14px;")
        self.study_n0_values_label = QLabel() # 修改: 创建实例
        params_layout.addRow(self.study_n0_values_label, self.study_n0_values_input)
        
        # 生成迭代次数
        self.study_gen_iterations_spin = QSpinBox()
        self.study_gen_iterations_spin.setRange(1, 5)
        self.study_gen_iterations_spin.setValue(3)
        self.study_gen_iterations_spin.setStyleSheet("font-size: 14px;")
        self.study_gen_iter_label = QLabel() # 修改: 创建实例
        params_layout.addRow(self.study_gen_iter_label, self.study_gen_iterations_spin)
        
        # 计算迭代次数
        self.study_calc_iterations_spin = QSpinBox()
        self.study_calc_iterations_spin.setRange(1, 8)
        self.study_calc_iterations_spin.setValue(3)
        self.study_calc_iterations_spin.setStyleSheet("font-size: 14px;")
        self.study_calc_iter_label = QLabel() # 修改: 创建实例
        params_layout.addRow(self.study_calc_iter_label, self.study_calc_iterations_spin)
        
        # 绑定2D-3D研究参数的生成迭代次数和计算迭代次数
        self.study_gen_iterations_spin.valueChanged.connect(lambda v: self._sync_spinbox_value('study_calc_iterations_spin', v))
        self.study_calc_iterations_spin.valueChanged.connect(lambda v: self._sync_spinbox_value('study_gen_iterations_spin', v))
        
        # 重复模拟次数
        self.study_repeat_count_spin = QSpinBox()
        self.study_repeat_count_spin.setRange(1, 100)
        self.study_repeat_count_spin.setValue(10)
        self.study_repeat_count_spin.setStyleSheet("font-size: 14px;")
        self.study_repeat_count_label = QLabel() # 修改: 创建实例
        params_layout.addRow(self.study_repeat_count_label, self.study_repeat_count_spin)

        self.study_num_slices_spin = QSpinBox()
        self.study_num_slices_spin.setRange(3, 201)
        self.study_num_slices_spin.setValue(21)
        self.study_num_slices_spin.setSingleStep(2)
        self.study_num_slices_spin.setStyleSheet("font-size: 14px;")
        self.study_num_slices_label = QLabel()
        params_layout.addRow(self.study_num_slices_label, self.study_num_slices_spin)
        
        self.study_params_group.setLayout(params_layout)
        layout.addWidget(self.study_params_group)
        
        # 并行计算设置组
        self.study_parallel_group = QGroupBox() # 修改: 移除文本
        self.study_parallel_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        parallel_layout = QFormLayout()
        
        self.study_enable_parallel_checkbox = QCheckBox()
        self.study_enable_parallel_checkbox.setChecked(True)
        self.study_enable_parallel_checkbox.setStyleSheet("font-size: 14px;")
        self.study_enable_parallel_checkbox.stateChanged.connect(self.on_parallel_mode_changed)
        self.study_parallel_mode_label = QLabel() # 修改: 创建实例
        parallel_layout.addRow(self.study_parallel_mode_label, self.study_enable_parallel_checkbox)
        
        # 进程数设置
        self.study_thread_count_spin = QSpinBox()
        self.study_thread_count_spin.setRange(1, 256)  # 最多256个进程（物理核心数）
        self.study_thread_count_spin.setValue(4)  # 默认4个进程
        self.study_thread_count_spin.setStyleSheet("font-size: 14px;")
        self.study_thread_count_label = QLabel() # 修改: 创建实例
        parallel_layout.addRow(self.study_thread_count_label, self.study_thread_count_spin)

        # 切片分布计算开关
        self.study_calc_profile_checkbox = QCheckBox()
        self.study_calc_profile_checkbox.setChecked(False) # 默认不勾选，节省空间
        self.study_calc_profile_checkbox.setStyleSheet("font-size: 14px;")
        self.study_calc_profile_label = QLabel() # 文本在 retranslate_ui 设置
        parallel_layout.addRow(self.study_calc_profile_label, self.study_calc_profile_checkbox)

        # === 新增：仅计算中心区域复选框 ===
        self.study_center_only_checkbox = QCheckBox()
        self.study_center_only_checkbox.setChecked(False)
        self.study_center_only_checkbox.setStyleSheet("font-size: 14px;")
        self.study_center_only_label = QLabel() # 文本在 retranslate_ui 设置
        parallel_layout.addRow(self.study_center_only_label, self.study_center_only_checkbox)
        # ================================
        
        self.study_parallel_group.setLayout(parallel_layout)
        layout.addWidget(self.study_parallel_group)
        
        # 控制按钮
        button_layout = QVBoxLayout()
        
        # 参数预估计算按钮
        self.study_param_estimate_btn = QPushButton()
        self.study_param_estimate_btn.clicked.connect(self.show_parameter_estimation)
        self.study_param_estimate_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        button_layout.addWidget(self.study_param_estimate_btn)
        
        self.study_start_calc_btn = QPushButton()
        self.study_start_calc_btn.clicked.connect(self.start_2d3d_calculation)
        self.study_start_calc_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.study_start_calc_btn)
        
        self.study_stop_calc_btn = QPushButton()
        self.study_stop_calc_btn.clicked.connect(self.stop_2d3d_calculation)
        self.study_stop_calc_btn.setEnabled(False)
        self.study_stop_calc_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.study_stop_calc_btn)
        
        self.study_clear_btn = QPushButton()
        self.study_clear_btn.clicked.connect(self.clear_2d3d_results)
        self.study_clear_btn.setEnabled(False)
        self.study_clear_btn.setStyleSheet("font-size: 16px;")
        button_layout.addWidget(self.study_clear_btn)
        
        self.study_load_backup_btn = QPushButton()
        self.study_load_backup_btn.clicked.connect(self.load_backup_file)
        self.study_load_backup_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        button_layout.addWidget(self.study_load_backup_btn)
        
        layout.addLayout(button_layout)
        
        # 进度条
        self.study_progress_bar = QProgressBar()
        self.study_progress_bar.setVisible(False)
        self.study_progress_bar.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.study_progress_bar)
        
        # 状态信息
        self.study_status_label = QLabel()
        self.study_status_label.setAlignment(Qt.AlignCenter)
        self.study_status_label.setWordWrap(True)
        self.study_status_label.setStyleSheet("color: #555; font-size: 14px; margin: 10px;")
        layout.addWidget(self.study_status_label)
        
        # 说明文本
        self.study_help_group = QGroupBox() # 修改: 移除文本
        self.study_help_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        help_layout = QVBoxLayout()
        
        
        self.study_help_text = QTextEdit()
        self.study_help_text.setReadOnly(True)
        self.study_help_text.setMaximumHeight(600)
        self.study_help_text.setStyleSheet("font-size: 12px; background-color: #f5f5f5;")

        help_layout.addWidget(self.study_help_text)
        self.study_help_group.setLayout(help_layout)
        layout.addWidget(self.study_help_group)
                
        panel.setLayout(layout)
        return panel
    
    def create_2d3d_results_area(self, parent):
        """创建2D-3D关系研究结果显示区域"""
        results_widget = QWidget()
        layout = QVBoxLayout()
        
        # 标题
        self.study_results_title_label = QLabel() # 修改
        self.study_results_title_label.setAlignment(Qt.AlignCenter)
        self.study_results_title_label.setStyleSheet("font-weight: bold; font-size: 18px; color: #2E86AB; margin: 10px;")
        layout.addWidget(self.study_results_title_label)
        
        # 显示模式选择区域
        mode_layout = QHBoxLayout()
        self.study_display_mode_label = QLabel() # 修改
        self.study_display_mode_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        mode_layout.addWidget(self.study_display_mode_label)
        
        self.study_display_mode_combo = QComboBox()
        # Items will be added in retranslate_ui
        self.study_display_mode_combo.currentTextChanged.connect(self.on_study_display_mode_changed)
        self.study_display_mode_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        mode_layout.addWidget(self.study_display_mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # 统计信息标签
        self.study_stats_label = QLabel() # 修改
        self.study_stats_label.setAlignment(Qt.AlignCenter)
        self.study_stats_label.setStyleSheet("font-size: 14px; color: #666; margin: 5px;")
        layout.addWidget(self.study_stats_label)
        
        # 创建结果表格
        self.study_results_table = QTableWidget()
        self.study_results_table.setAlternatingRowColors(True)
        self.study_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.study_results_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 5px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.study_results_table)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.study_copy_btn = QPushButton() # 修改
        self.study_copy_btn.clicked.connect(self.copy_2d3d_results)
        self.study_copy_btn.setEnabled(False)
        self.study_copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.study_copy_btn)
        
        self.study_save_btn = QPushButton() # 修改
        self.study_save_btn.clicked.connect(self.save_2d3d_results)
        self.study_save_btn.setEnabled(False)
        self.study_save_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.study_save_btn)
        
        self.study_export_raw_btn = QPushButton() # 修改
        self.study_export_raw_btn.clicked.connect(self.export_raw_2d3d_results)
        self.study_export_raw_btn.setEnabled(False)
        self.study_export_raw_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.study_export_raw_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        results_widget.setLayout(layout)
        return results_widget
    
    def on_parallel_mode_changed(self):
        """并行模式切换回调"""
        is_parallel = self.study_enable_parallel_checkbox.isChecked()
        self.study_thread_count_spin.setEnabled(is_parallel)
        
        if is_parallel:
            self.study_status_label.setText("已启用并行计算模式，将使用多进程加速计算")
        else:
            self.study_status_label.setText("已切换到单线程模式，计算将按顺序进行")
    
    def on_study_display_mode_changed(self):
        """显示模式切换回调"""
        # --- 核心修改：不再依赖文本，但为了调试方便可以保留 current_display_mode ---
        self.current_display_mode = self.study_display_mode_combo.currentText()
        # 重新刷新显示
        self.update_2d3d_results_display()
    
# 文件: FD-Advanced_3D.py

    def _aggregate_study_results(self):
        """聚合研究结果，计算相同参数组合的平均值和标准差"""
        if not self.study_results:
            return []
        
        groups = {}
        for result in self.study_results:
            key = (result['cube_size'], result['theoretical_dim'], result['theoretical_n0'])
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        aggregated = []
        for key, group_results in groups.items():
            cube_size, theoretical_dim, theoretical_n0 = key
            
            # 提取所有有效数据
            actual_3d_dims = [r['actual_3d_dim'] for r in group_results if r.get('actual_3d_dim') is not None]
            actual_3d_n0s = [r['actual_3d_n0'] for r in group_results if r.get('actual_3d_n0') is not None]

            # --- 新增：提取所有新的2D数据 ---
            average_2d_dims_all = [r['average_2d_dim_all'] for r in group_results if r.get('average_2d_dim_all') is not None]
            average_2d_n0s_all = [r['average_2d_n0_all'] for r in group_results if r.get('average_2d_n0_all') is not None]
            average_2d_dims_yoz = [r['average_2d_dim_yoz'] for r in group_results if r.get('average_2d_dim_yoz') is not None]
            average_2d_n0s_yoz = [r['average_2d_n0_yoz'] for r in group_results if r.get('average_2d_n0_yoz') is not None]
            average_2d_dims_xoz = [r['average_2d_dim_xoz'] for r in group_results if r.get('average_2d_dim_xoz') is not None]
            average_2d_n0s_xoz = [r['average_2d_n0_xoz'] for r in group_results if r.get('average_2d_n0_xoz') is not None]
            average_2d_dims_xoy = [r['average_2d_dim_xoy'] for r in group_results if r.get('average_2d_dim_xoy') is not None]
            average_2d_n0s_xoy = [r['average_2d_n0_xoy'] for r in group_results if r.get('average_2d_n0_xoy') is not None]
            cuboid_unfolding_dims = [r['cuboid_unfolding_dim'] for r in group_results if r.get('cuboid_unfolding_dim') is not None]
            cuboid_unfolding_n0s = [r['cuboid_unfolding_n0'] for r in group_results if r.get('cuboid_unfolding_n0') is not None]
            cylinder_unfolding_dims = [r['cylinder_unfolding_dim'] for r in group_results if r.get('cylinder_unfolding_dim') is not None]
            cylinder_unfolding_n0s = [r['cylinder_unfolding_n0'] for r in group_results if r.get('cylinder_unfolding_n0') is not None]
            
            def calc_stats(values):
                if not values:
                    return None, None
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0
                return mean_val, std_val


            # === 新增：聚合分布曲线数据 ===
            # 1. 提取所有有效的分布列表
            valid_profile_dims = []
            valid_profile_n0s = []
            
            for r in group_results:
                # 兼容性处理：数据库加载的数据可能是 JSON 字符串，内存中的是 List
                p_dim = r.get('slice_profile_dim')
                p_n0 = r.get('slice_profile_n0')
                
                if isinstance(p_dim, str): p_dim = json.loads(p_dim)
                if isinstance(p_n0, str): p_n0 = json.loads(p_n0)
                
                if p_dim and isinstance(p_dim, list): valid_profile_dims.append(p_dim)
                if p_n0 and isinstance(p_n0, list): valid_profile_n0s.append(p_n0)
            
            # 2. 计算按位平均 (Element-wise Mean)
            # 注意：列表中可能包含 None (如果某次切片没切到东西)，需要用 np.nanmean 处理
            
            def calc_profile_mean(profiles_list):
                if not profiles_list: return None
                try:
                    # 转换为 numpy 数组，将 None 视为 np.nan
                    arr = np.array(profiles_list, dtype=float)
                    # 计算列平均 (忽略 NaN)
                    mean_arr = np.nanmean(arr, axis=0)
                    # 将 NaN 替换回 None 或 0，转为 list
                    return [None if np.isnan(x) else x for x in mean_arr]
                except Exception as e:
                    print(f"聚合分布数据出错: {e}")
                    return None

            profile_dim_mean = calc_profile_mean(valid_profile_dims)
            profile_n0_mean = calc_profile_mean(valid_profile_n0s)


            actual_3d_dim_mean, actual_3d_dim_std = calc_stats(actual_3d_dims)
            actual_3d_n0_mean, actual_3d_n0_std = calc_stats(actual_3d_n0s)
            
            # --- 新增：计算所有新2D数据的统计值 ---
            average_2d_dim_all_mean, average_2d_dim_all_std = calc_stats(average_2d_dims_all)
            average_2d_n0_all_mean, average_2d_n0_all_std = calc_stats(average_2d_n0s_all)
            average_2d_dim_yoz_mean, average_2d_dim_yoz_std = calc_stats(average_2d_dims_yoz)
            average_2d_n0_yoz_mean, average_2d_n0_yoz_std = calc_stats(average_2d_n0s_yoz)
            average_2d_dim_xoz_mean, average_2d_dim_xoz_std = calc_stats(average_2d_dims_xoz)
            average_2d_n0_xoz_mean, average_2d_n0_xoz_std = calc_stats(average_2d_n0s_xoz)
            average_2d_dim_xoy_mean, average_2d_dim_xoy_std = calc_stats(average_2d_dims_xoy)
            average_2d_n0_xoy_mean, average_2d_n0_xoy_std = calc_stats(average_2d_n0s_xoy)
            cuboid_unfolding_dim_mean, cuboid_unfolding_dim_std = calc_stats(cuboid_unfolding_dims)
            cuboid_unfolding_n0_mean, cuboid_unfolding_n0_std = calc_stats(cuboid_unfolding_n0s)
            cylinder_unfolding_dim_mean, cylinder_unfolding_dim_std = calc_stats(cylinder_unfolding_dims)
            cylinder_unfolding_n0_mean, cylinder_unfolding_n0_std = calc_stats(cylinder_unfolding_n0s)

            aggregated.append({
                'cube_size': cube_size,
                'theoretical_dim': theoretical_dim,
                'theoretical_n0': theoretical_n0,
                'actual_3d_dim_mean': actual_3d_dim_mean,
                'actual_3d_dim_std': actual_3d_dim_std,
                'actual_3d_n0_mean': actual_3d_n0_mean,
                'actual_3d_n0_std': actual_3d_n0_std,

                # --- 新增：将所有新统计值添加到聚合结果中 ---
                'average_2d_dim_all_mean': average_2d_dim_all_mean, 'average_2d_dim_all_std': average_2d_dim_all_std,
                'average_2d_n0_all_mean': average_2d_n0_all_mean, 'average_2d_n0_all_std': average_2d_n0_all_std,
                'average_2d_dim_yoz_mean': average_2d_dim_yoz_mean, 'average_2d_dim_yoz_std': average_2d_dim_yoz_std,
                'average_2d_n0_yoz_mean': average_2d_n0_yoz_mean, 'average_2d_n0_yoz_std': average_2d_n0_yoz_std,
                'average_2d_dim_xoz_mean': average_2d_dim_xoz_mean, 'average_2d_dim_xoz_std': average_2d_dim_xoz_std,
                'average_2d_n0_xoz_mean': average_2d_n0_xoz_mean, 'average_2d_n0_xoz_std': average_2d_n0_xoz_std,
                'average_2d_dim_xoy_mean': average_2d_dim_xoy_mean, 'average_2d_dim_xoy_std': average_2d_dim_xoy_std,
                'average_2d_n0_xoy_mean': average_2d_n0_xoy_mean, 'average_2d_n0_xoy_std': average_2d_n0_xoy_std,
                'cuboid_unfolding_dim_mean': cuboid_unfolding_dim_mean, 'cuboid_unfolding_dim_std': cuboid_unfolding_dim_std,
                'cuboid_unfolding_n0_mean': cuboid_unfolding_n0_mean, 'cuboid_unfolding_n0_std': cuboid_unfolding_n0_std,
                'cylinder_unfolding_dim_mean': cylinder_unfolding_dim_mean, 'cylinder_unfolding_dim_std': cylinder_unfolding_dim_std,
                'cylinder_unfolding_n0_mean': cylinder_unfolding_n0_mean, 'cylinder_unfolding_n0_std': cylinder_unfolding_n0_std,
                
                'slice_profile_dim_mean': profile_dim_mean,
                'slice_profile_n0_mean': profile_n0_mean,

                'repeat_count': len(group_results)
            })
        
        aggregated.sort(key=lambda x: (x['cube_size'], x['theoretical_dim'], x['theoretical_n0']))
        return aggregated
    
    def show_parameter_estimation(self):
        """显示参数预估计算结果"""
        try:
            # 获取参数输入
            cube_lengths = self.parse_parameter(self.study_cube_lengths_input.text(), "立方体长度")
            fractal_dims = self.parse_parameter(self.study_fractal_dims_input.text(), "分形维数")
            n0_values = self.parse_parameter(self.study_n0_values_input.text(), "分形初值")
            calc_iterations = self.study_calc_iterations_spin.value()
            
            if cube_lengths is None or fractal_dims is None or n0_values is None:
                return
            
            if not cube_lengths or not fractal_dims or not n0_values:
                QMessageBox.warning(self, self.tr("警告"), self.tr("请输入有效的参数值"))
                return
            
            # 计算所有参数组合和所有迭代盒子尺寸的N值（包含初始层级）
            results_by_box_size = {}
            
            for L in cube_lengths:
                for D in fractal_dims:
                    for N0 in n0_values:
                        # 计算每个迭代层级的盒子尺寸和对应的N值（包含第0次迭代）
                        for iteration in range(calc_iterations + 1):  # +1 包含初始层级
                            # 盒子尺寸计算：L / (2^iteration)
                            box_size = L / (2 ** iteration)
                            N = N0 * (box_size ** (-D))
                            
                            # 按盒子尺寸分组存储结果
                            box_key = f"{box_size:.6f}"
                            if box_key not in results_by_box_size:
                                results_by_box_size[box_key] = {
                                    'box_size': box_size,
                                    'iteration': iteration,
                                    'results': []
                                }
                            
                            results_by_box_size[box_key]['results'].append({
                                'L': L,
                                'D': D,
                                'N0': N0,
                                'box_size': box_size,
                                'iteration': iteration,
                                'N': N
                            })
            
            # 创建结果显示窗口
            self.show_estimation_results_with_tabs(results_by_box_size, calc_iterations)
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr(f"参数预估计算失败: {str(e)}"))
    

    
    def copy_estimation_results(self, results):
        """复制参数预估计算结果到剪贴板"""
        try:
            # 构建表格文本
            text_lines = []
            text_lines.append("参数预估计算结果")
            text_lines.append("计算公式: N = N0 × L^(-D)")
            text_lines.append("")
            text_lines.append("立方体长度(L)\t分形维数(D)\t分形初值(N0)\t计算结果(N)")
            
            for result in results:
                line = f"{result['L']}\t{result['D']}\t{result['N0']}\t{result['N']:.6f}"
                text_lines.append(line)
            
            # 复制到剪贴板
            clipboard = QApplication.clipboard()
            clipboard.setText('\n'.join(text_lines))
            
            QMessageBox.information(self, self.tr("成功"), self.tr("结果已复制到剪贴板"))
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr(f"复制失败: {str(e)}"))
    
    def show_estimation_results_with_tabs(self, results_by_box_size, calc_iterations):
        """使用标签页显示不同盒子尺寸的参数预估计算结果"""
        dialog = QDialog(self)
        dialog.setWindowTitle("参数预估计算结果 - 按盒子尺寸分组")
        dialog.setModal(True)
        dialog.resize(1000, 700)
        
        layout = QVBoxLayout(dialog)
        
        # 标题
        title_label = QLabel(f"参数预估计算结果 (共 {calc_iterations + 1} 个迭代层级)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title_label)
        
        # 创建标签页控件
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #34495e;
                background-color: white;
                border-radius: 5px;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                color: #2c3e50;
                padding: 10px 18px;
                margin-right: 3px;
                border: 2px solid #bdc3c7;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
                border-color: #34495e;
                border-bottom: 2px solid #3498db;
            }
            QTabBar::tab:hover {
                background-color: #d5dbdb;
                border-color: #95a5a6;
            }
        """)
        
        # 按盒子尺寸排序
        sorted_box_sizes = sorted(results_by_box_size.items(), key=lambda x: x[1]['box_size'], reverse=True)
        
        # 为每个盒子尺寸创建标签页
        for box_key, box_data in sorted_box_sizes:
            box_size = box_data['box_size']
            iteration = box_data['iteration']
            results = box_data['results']
            
            # 创建标签页
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # 盒子信息标签
            info_label = QLabel(f"盒子尺寸: {box_size:.6f} | 迭代次数: {iteration} | 参数组合数: {len(results)}")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #7f8c8d;
                    padding: 5px;
                    background-color: #f8f9fa;
                    border-radius: 3px;
                    margin-bottom: 10px;
                }
            """)
            tab_layout.addWidget(info_label)
            
            # 结果表格
            table = QTableWidget()
            table.setRowCount(len(results))
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(["立方体长度 (L)", "分形维数 (D)", "分形初值 (N0)", "盒子尺寸", "计算结果 (N)"])
            
            # 设置表格样式
            table.setStyleSheet("""
                QTableWidget {
                    gridline-color: #bdc3c7;
                    background-color: white;
                    alternate-background-color: #f8f9fa;
                    font-size: 12px;
                }
                QTableWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #ecf0f1;
                    font-size: 12px;
                }
                QHeaderView::section {
                    background-color: #34495e;
                    color: white;
                    padding: 10px;
                    border: none;
                    font-weight: bold;
                    font-size: 12px;
                }
            """)
            
            table.setAlternatingRowColors(True)
            table.horizontalHeader().setStretchLastSection(True)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            
            # 填充数据
            for i, result in enumerate(results):
                table.setItem(i, 0, QTableWidgetItem(f"{result['L']:.3f}"))
                table.setItem(i, 1, QTableWidgetItem(f"{result['D']:.3f}"))
                table.setItem(i, 2, QTableWidgetItem(f"{result['N0']:.3f}"))
                table.setItem(i, 3, QTableWidgetItem(f"{result['box_size']:.3f}"))
                table.setItem(i, 4, QTableWidgetItem(f"{result['N']:.3f}"))
            
            tab_layout.addWidget(table)
            
            # 为每个标签页添加复制按钮
            copy_button = QPushButton(f"复制盒子尺寸 {box_size:.6f} 的结果")
            copy_button.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #229954;
                }
                QPushButton:pressed {
                    background-color: #1e8449;
                }
            """)
            copy_button.clicked.connect(lambda checked, r=results: self.copy_estimation_results(r))
            tab_layout.addWidget(copy_button)
            
            # 添加标签页
            tab_name = f"尺寸 {box_size:.4f} (第{iteration}次)"
            tab_widget.addTab(tab, tab_name)
        
        layout.addWidget(tab_widget)
        
        # 底部按钮区域
        button_layout = QHBoxLayout()
        
        # 复制所有结果按钮
        copy_all_button = QPushButton("复制所有结果")
        copy_all_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        all_results = []
        for box_data in results_by_box_size.values():
            all_results.extend(box_data['results'])
        copy_all_button.clicked.connect(lambda: self.copy_estimation_results(all_results))
        button_layout.addWidget(copy_all_button)
        
        # 关闭按钮
        close_button = QPushButton("关闭")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:pressed {
                background-color: #6c7b7d;
            }
        """)
        close_button.clicked.connect(dialog.close)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()
    
    def copy_estimation_results(self, results):
        """复制参数预估计算结果到剪贴板"""
        try:
            # 构建表格文本
            text_lines = []
            text_lines.append("参数预估计算结果")
            text_lines.append("计算公式: N = N0 × L^(-D)")
            text_lines.append("")
            text_lines.append("立方体长度(L)\t分形维数(D)\t分形初值(N0)\t计算结果(N)")
            
            for result in results:
                line = f"{result['L']}\t{result['D']}\t{result['N0']}\t{result['N']:.6f}"
                text_lines.append(line)
            
            # 复制到剪贴板
            clipboard = QApplication.clipboard()
            clipboard.setText('\n'.join(text_lines))
            
            QMessageBox.information(self, self.tr("成功"), self.tr("结果已复制到剪贴板"))
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr(f"复制失败: {str(e)}"))
    
    def parse_parameter(self, text, param_name="参数"):
        """
        解析参数输入，支持三种格式：
        1. 单个值: e.g., "2.5"
        2. 空格分隔: e.g., "10 20 30"
        3. (起点, 终点, 步长) 序列格式: e.g., "(2.0, 3.0, 0.1)"
        -> 从2.0到3.0，步长0.1，包含起点和终点。
        """
        text = text.strip()
        if not text:
            return []
            
        try:
            # 检查是否为 (起点, 终点, 步长) 格式
            match = re.match(r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text)
            if match:
                start, end, step = map(float, match.groups())
                
                # --- 参数验证 ---
                if step <= 0:
                    QMessageBox.warning(self, self.tr("警告"), self.tr(f"{param_name}的步长必须大于0"))
                    return None
                if start > end:
                    QMessageBox.warning(self, self.tr("警告"), self.tr(f"{param_name}的起始值不能大于结束值"))
                    return None
                
                # --- 核心逻辑 ---
                # np.arange的终点是开区间，为了包含终点，我们将stop值设置得比end稍大一点。
                # 这可以巧妙地解决浮点数精度问题，并确保如果 end 恰好是序列中的一个点，它会被包含。
                stop_value = end + step * 0.5 
                
                values = np.arange(start, stop_value, step)
                
                # 确保即使由于精度问题，最后一个值略大于end，也将其钳制在end
                if values.size > 0 and values[-1] > end:
                    values[-1] = end

                return [round(v, 6) for v in values] # 四舍五入到6位小数

            # 检查是否为空格分隔
            if ' ' in text or '\t' in text:
                values = []
                for val in text.split():
                    val = val.strip()
                    if val:
                        values.append(float(val))
                return values
                
            # 单个值
            return [float(text)]
            
        except ValueError as e:
            QMessageBox.warning(self, self.tr("警告"), self.tr(f"{param_name}格式错误: {str(e)}"))
            return None
    
# 这是最终的、完整的函数。请用它替换您代码中的整个同名函数。

# 文件: FD-Advanced_3D.py (替换 start_2d3d_calculation)

# 文件: FD-Advanced_3D.py (替换 start_2d3d_calculation)

# 文件: FD-Advanced_3D.py

    def start_2d3d_calculation(self):
        """开始2D-3D关系研究计算（修复参数传递顺序版）"""
        if self.study_2d3d_thread is not None and self.study_2d3d_thread.isRunning():
            QMessageBox.warning(self, self.tr("警告"), self.tr("计算正在进行中，请等待完成"))
            return
        
        try:
            # 步骤1: 解析分形参数
            cube_lengths = self.parse_parameter(self.study_cube_lengths_input.text(), "立方体长度")
            fractal_dims = self.parse_parameter(self.study_fractal_dims_input.text(), "分形维数")
            n0_values = self.parse_parameter(self.study_n0_values_input.text(), "分形初值")
            
            if cube_lengths is None or fractal_dims is None or n0_values is None: return
            if not cube_lengths or not fractal_dims or not n0_values:
                QMessageBox.warning(self, self.tr("警告"), self.tr("请输入有效的参数值"))
                return
                
            total_combinations = len(cube_lengths) * len(fractal_dims) * len(n0_values)
            repeat_count = self.study_repeat_count_spin.value()
            total_simulations = total_combinations * repeat_count
            enable_parallel = self.study_enable_parallel_checkbox.isChecked()
            thread_count = self.study_thread_count_spin.value() if enable_parallel else 1
            mode_text = self.tr("并行模式({0}进程)").format(thread_count) if enable_parallel else self.tr("单线程模式")

            # 步骤2: 弹窗确认
            message_template = self.tr(
                "即将开始一次全新的计算任务。\n\n"
                "计算模式：{0}\n"
                "参数组合数：{1} 组\n"
                "每组重复次数：{2} 次\n"
                "----------------------------------\n"
                "总模拟次数：{3} 次\n\n"
                "注意：如果当前存在结果，它们将被清空。\n\n"
                "是否确认开始？"
            )
            confirm_msg = message_template.format(mode_text, total_combinations, repeat_count, total_simulations)
            
            if QMessageBox.question(self, self.tr("确认开始计算"), confirm_msg) != QMessageBox.Yes:
                return

            # --- 用户确认后 ---
            self.clear_2d3d_results()
            self.completed_tasks_offset = 0

            # 步骤3: 获取所有裂缝形态参数
            random_aspect_ratio = self.random_aspect_ratio_check.isChecked()
            aspect_ratio_base = self.aspect_ratio_base_spin.value()
            aspect_ratio_variation = self.aspect_ratio_variation_spin.value()
            is_isotropic = self.isotropic_check.isChecked()
            mean_inclination = self.mean_inclination_spin.value()
            mean_azimuth = self.mean_azimuth_spin.value()
            kappa = self.kappa_spin.value()
            use_advanced_model = self.advanced_model_check.isChecked()

            gen_iterations = self.study_gen_iterations_spin.value()
            calc_iterations = self.study_calc_iterations_spin.value()
            num_slices = self.study_num_slices_spin.value()
            calc_slice_profile = self.study_calc_profile_checkbox.isChecked() # 获取切片分析开关
            calc_center_only = self.study_center_only_checkbox.isChecked()

            # 步骤4: 准备UI和备份
            self.study_start_calc_btn.setEnabled(False)
            self.study_progress_bar.setVisible(True)
            self.study_progress_bar.setRange(0, total_simulations)
            self._initialize_results_table()
            
            gui_input_params = {
                "cube_lengths_str": self.study_cube_lengths_input.text(),
                "fractal_dims_str": self.study_fractal_dims_input.text(),
                "n0_values_str": self.study_n0_values_input.text(),
                "repeat_count": repeat_count,
                "gen_iterations": gen_iterations,
                "calc_iterations": calc_iterations,
                "num_slices": num_slices,
                "calc_slice_profile": calc_slice_profile, # 存入参数
                "calc_center_only": calc_center_only,
                "enable_parallel": enable_parallel,
                "max_processes": thread_count
            }
            
            from datetime import datetime
            import os
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            backup_dir = os.path.join(current_script_dir, "FD-Advanced_3D_temp_data")
            os.makedirs(backup_dir, exist_ok=True)
            backup_file_path = os.path.join(backup_dir, f"fractal_calculation_backup_{timestamp}.db")
            self.current_backup_file = backup_file_path
            self.current_session_timestamp = timestamp

            # 步骤5: 创建并启动后台线程
            if enable_parallel:
                self.study_2d3d_thread = Study2D3DParallelCalculationThread(
                    # 位置参数 1-16 (必须与 __init__ 严格对应)
                    cube_lengths,           # 1
                    fractal_dims,           # 2
                    n0_values,              # 3
                    repeat_count,           # 4
                    gen_iterations,         # 5
                    calc_iterations,        # 6
                    num_slices,             # 7
                    calc_slice_profile,     # 8
                    calc_center_only, # <--- 传递参数 (第9位)
                    
                    random_aspect_ratio,    # 9
                    aspect_ratio_base,      # 10
                    aspect_ratio_variation, # 11
                    is_isotropic,           # 12
                    mean_inclination,       # 13
                    mean_azimuth,           # 14
                    kappa,                  # 15
                    use_advanced_model,     # 16 (新增)
                    self.results_queue,     # 17

                    
                    # 关键字参数
                    max_processes=thread_count, 
                    backup_file_path=backup_file_path,
                    gui_input_params=gui_input_params
                )
            else: # 单线程模式
                self.study_2d3d_thread = Study2D3DCalculationThread(
                    # 位置参数 1-15
                    cube_lengths,           # 1
                    fractal_dims,           # 2
                    n0_values,              # 3
                    repeat_count,           # 4
                    gen_iterations,         # 5
                    calc_iterations,        # 6
                    num_slices,             # 7
                    calc_slice_profile,     # 8
                    calc_center_only, # <--- 传递参数 (第9位)
                    
                    random_aspect_ratio,    # 9
                    aspect_ratio_base,      # 10
                    aspect_ratio_variation, # 11
                    is_isotropic,           # 12
                    mean_inclination,       # 13
                    mean_azimuth,           # 14
                    kappa,                  # 15
                    use_advanced_model,     # 16 (新增)
                    
                    # 关键字参数
                    backup_file_path=backup_file_path, 
                    gui_input_params=gui_input_params
                )
            
            self.study_2d3d_thread.calculation_finished.connect(self.on_study_calculation_finished)
            self.study_2d3d_thread.calculation_failed.connect(self.on_study_calculation_failed)
            
            if not enable_parallel:
                self.study_2d3d_thread.single_result_ready.connect(self.result_buffer.append)
            
            self.study_2d3d_thread.start()
            self.study_stop_calc_btn.setEnabled(True)
            self.ui_update_timer.start()

            status_text = self.tr("计算进行中... ({0})\n备份文件: {1}").format(mode_text, backup_file_path)
            self.study_status_label.setText(status_text)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, self.tr("错误"), self.tr("启动计算时发生错误:\n{0}").format(error_msg))
            self.study_start_calc_btn.setEnabled(True)
            self.study_stop_calc_btn.setEnabled(False)
            self.study_progress_bar.setVisible(False)
    
# 文件: FD-Advanced_3D.py (替换 stop_2d3d_calculation)

# 文件: FD-Advanced_3D.py (替换 stop_2d3d_calculation)

    def stop_2d3d_calculation(self):
        """停止2D-3D关系研究计算（最终版，确保所有活动结束后再提示）"""
        if self.study_2d3d_thread and self.study_2d3d_thread.isRunning():
            # 1. 立即更新UI，提供即时反馈，并防止重复点击
            self.study_stop_calc_btn.setEnabled(False)
            self.study_start_calc_btn.setEnabled(False) # 停止期间，开始按钮也应禁用
            self.study_status_label.setText(self.tr("正在停止，请稍候... (等待所有子进程结束)"))
            QApplication.processEvents()

            # 2. 发出停止信号，并等待主计算线程和所有子进程完全退出
            self.study_2d3d_thread.stop_calculation()
            
            # 等待最多10秒，对于写盘等操作给予充足时间
            if not self.study_2d3d_thread.wait(10000):
                print("线程未能优雅退出，将强制终止。")
                self.study_2d3d_thread.terminate()
                self.study_2d3d_thread.wait()
            
            # 3. 线程已结束，清理线程对象
            self.study_2d3d_thread.deleteLater()
            self.study_2d3d_thread = None

            # <<< 核心修改点：在后台线程确认停止后执行 >>>
            self.ui_update_timer.stop()
            self._process_result_buffer_and_update_ui()
            # <<< 修改结束 >>>

            # 4. 更新状态栏，告知用户正在进行最后的数据同步
            self.study_status_label.setText(self.tr("所有进程已停止。正在从备份文件同步最终结果..."))
            QApplication.processEvents()

            # 5. 从DB文件重新加载，确保数据最终一致性
            try:
                if self.current_backup_file and os.path.exists(self.current_backup_file):
                    print(f"正在从 {self.current_backup_file} 重新加载数据以确保一致性...")
                    backup_manager = SQLiteBackupManager(self.current_backup_file)
                    backup_data = backup_manager.load_backup_data()
                    if backup_data and 'results' in backup_data:
                        self.study_results = backup_data['results']
                        print(f"同步成功，最终加载了 {len(self.study_results)} 条结果。")
                        # 刷新整个结果显示区域（表格和统计信息）
                        self.update_2d3d_results_display()
                else:
                    print("未找到有效的备份文件，无法同步最终数据。")

            except Exception as e:
                import traceback
                print(f"停止后同步数据失败: {e}\n{traceback.format_exc()}")
                QMessageBox.warning(self, self.tr("警告"), self.tr("从备份文件同步最终结果时发生错误。"))

            # 6. 根据最终同步后的结果，更新UI状态
            self.study_progress_bar.setVisible(False)
            if self.study_results:
                self.study_clear_btn.setEnabled(True)
                status_text = self.tr("计算已停止。最终获得 {0} 条数据。").format(len(self.study_results))
                self.study_status_label.setText(status_text)
            else:
                self.study_clear_btn.setEnabled(False)
                self.study_status_label.setText(self.tr("计算已停止。"))
            
            # 重新启用开始按钮
            self.study_start_calc_btn.setEnabled(True)

            # 7. 在所有操作完成后，给出最终的成功提示
            QMessageBox.information(self, self.tr("操作完成"), self.tr("计算已成功停止，并已同步所有已保存的结果。"))
    
    def _process_result_buffer_and_update_ui(self):
        """
        (新函数) 由QTimer触发，从缓冲区批量处理结果并更新UI。
        """
        results_to_process = []
        # 为避免单次更新任务过重导致界面卡顿，设置一个处理上限
        max_items_per_update = 500
        for _ in range(max_items_per_update):
            try:
                # 从缓冲区左侧弹出一个结果
                results_to_process.append(self.result_buffer.popleft())
            except IndexError:
                # 当缓冲区为空时，popleft会引发IndexError，表示处理完毕
                break
        
        # 如果没有需要处理的结果，直接返回
        if not results_to_process:
            return

        # 将这批新结果批量添加到核心数据列表中
        self.study_results.extend(results_to_process)
        
        # 对表格进行高效的批量更新
        self.study_results_table.setUpdatesEnabled(False) # 关键优化：暂停UI重绘，避免每次加行都重绘
        try:
            for result in results_to_process:
                # 调用旧的行添加逻辑，但此时UI不会响应
                self._add_single_result_to_table(result) 
        finally:
            self.study_results_table.setUpdatesEnabled(True) # 关键优化：恢复UI重绘，所有新行一次性显示
        
        self.study_results_table.scrollToBottom() # 所有行添加完后只滚动一次

        # 一次性更新进度条和状态文本
        current_progress = len(self.study_results)
        total_simulations = self.study_progress_bar.maximum()
        self.study_progress_bar.setValue(current_progress)
        
        status_text = self.tr("计算进行中... 已完成 {0}/{1}").format(current_progress, total_simulations)
        self.study_status_label.setText(status_text)
    

    
# --- START OF REPLACEMENT CODE for on_study_calculation_finished() ---
    def on_study_calculation_finished(self, status_code):
        """处理计算完成信号 (增加了定时器停止和最终刷新逻辑)"""
        
        # <<< 核心修改点：在函数最开始执行 >>>
        # 1. 停止UI更新定时器
        self.ui_update_timer.stop()
        # 2. 手动调用一次批量处理函数，以确保缓冲区中所有剩余的结果都被处理和显示
        self._process_result_buffer_and_update_ui() 
        # <<< 修改结束 >>>

        self.study_progress_bar.setVisible(False)
        self.study_start_calc_btn.setEnabled(True)
        self.study_stop_calc_btn.setEnabled(False)
        
        # status_code == -1 表示是用户中止的
        if status_code == -1:
            status_text = self.tr("计算已中止。共获得 {0} 条数据").format(len(self.study_results))
        else:
            status_text = self.tr("计算完成！共获得 {0} 条数据").format(len(self.study_results))
        
        self.study_status_label.setText(status_text)
        
        self.update_2d3d_results_display()
        
        # 线程使命已完成，现在可以安全地安排销毁
        if self.study_2d3d_thread:
            self.study_2d3d_thread.deleteLater()
            self.study_2d3d_thread = None
    # --- END OF REPLACEMENT CODE ---
    
    # 请用这个新的、完整的函数替换您代码中的同名函数

    def on_study_calculation_failed(self, error_message):
        """处理计算失败"""

        # <<< 核心修改点：在函数最开始执行 >>>
        self.ui_update_timer.stop()
        self._process_result_buffer_and_update_ui()
        # <<< 修改结束 >>>

        self.study_progress_bar.setVisible(False)
        self.study_start_calc_btn.setEnabled(True)
        self.study_stop_calc_btn.setEnabled(False)
        
        if self.study_results:
            # --- 核心修改 ---
            status_text = self.tr("计算中断，已获得 {0} 条数据").format(len(self.study_results))
            self.study_status_label.setText(status_text)
            
            message_template = self.tr("计算过程中发生错误，但已保存 {0} 条有效结果:\n{1}")
            final_message = message_template.format(len(self.study_results), error_message)
            QMessageBox.warning(self, self.tr("计算中断"), final_message)

            self.update_2d3d_results_display()
        else:
            self.study_status_label.setText(self.tr("计算失败"))
            final_message = self.tr("计算过程中发生错误:\n{0}").format(error_message)
            QMessageBox.critical(self, self.tr("计算错误"), final_message)
        
        if self.study_2d3d_thread:
            self.study_2d3d_thread.deleteLater()
            self.study_2d3d_thread = None
    

    # 这是最终的、正确的版本。请用它完整替换文件中的同名函数。

    def _initialize_results_table(self):
        """初始化结果表格"""
        # --- 修改：定义新的、更宽的列结构 ---
        columns = [
            self.tr('立方体边长'), self.tr('理论D'), self.tr('理论N0'), 
            self.tr('实际3D D'), self.tr('实际3D N0'), 
            self.tr('总平均2D D'), self.tr('总平均2D N0'),
            self.tr('YOZ平均2D D'), self.tr('YOZ平均2D N0'),
            self.tr('XOZ平均2D D'), self.tr('XOZ平均2D N0'),
            self.tr('XOY平均2D D'), self.tr('XOY平均2D N0'),
            self.tr('展开-立方体D'), self.tr('展开-立方体N0'),
            self.tr('展开-圆柱体D'), self.tr('展开-圆柱体N0'),

        ]
        # === 动态添加列 ===
        # 检查复选框状态来决定是否显示列
        if self.study_calc_profile_checkbox.isChecked():
            columns.append(self.tr('位置趋势分析'))
        # =================
        
        columns.append(self.tr('重复序号')) # 最后一列


        self.study_results_table.setColumnCount(len(columns))
        self.study_results_table.setHorizontalHeaderLabels(columns)
        self.study_results_table.setRowCount(0)
        
        self.study_stats_label.setText(self.tr("计算中..."))
        
        self.study_copy_btn.setEnabled(True)
        self.study_save_btn.setEnabled(False)
        self.study_clear_btn.setEnabled(False)
        self.study_export_raw_btn.setEnabled(False)
    
    def _add_single_result_to_table(self, result):
        """添加单个结果到表格"""
        current_row = self.study_results_table.rowCount()
        self.study_results_table.insertRow(current_row)
        
        # 填充数据
        col = 0
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('cube_size', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('theoretical_dim', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('theoretical_n0', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('actual_3d_dim', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('actual_3d_n0', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('average_2d_dim_all', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('average_2d_n0_all', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('average_2d_dim_yoz', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('average_2d_n0_yoz', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('average_2d_dim_xoz', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('average_2d_n0_xoz', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('average_2d_dim_xoy', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('average_2d_n0_xoy', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('cuboid_unfolding_dim', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('cuboid_unfolding_n0', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('cylinder_unfolding_dim', 'N/A')))); col+=1
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('cylinder_unfolding_n0', 'N/A')))); col+=1
        
        self.study_results_table.setItem(current_row, col, QTableWidgetItem(str(result.get('repeat_index', 'N/A')))); col+=1
        self.study_results_table.resizeColumnsToContents()
        

        
        # 确保复制按钮始终可用，但保存按钮在计算过程中保持禁用
        self.study_copy_btn.setEnabled(True)
        # 注意：这里不启用保存按钮，因为计算还在进行中
    

    
    def update_2d3d_results_display(self):
        """更新结果显示"""
        if not self.study_results:
            self.study_stats_label.setText(self.tr("暂无数据"))
            self.study_copy_btn.setEnabled(True)
            self.study_save_btn.setEnabled(False)
            self.study_export_raw_btn.setEnabled(False)
            self.study_clear_btn.setEnabled(False)
            return
        
        # --- 核心修复：使用索引进行逻辑判断 ---
        current_index = self.study_display_mode_combo.currentIndex()
        
        if current_index == 0:  # 索引0代表“统计汇总”
            self._display_aggregated_results()
        else:  # 索引1 (或其他) 代表“原始数据”
            self._display_raw_results()
        
        # 启用按钮 (这部分逻辑保持不变)
        self.study_copy_btn.setEnabled(True)
        self.study_save_btn.setEnabled(True)
        self.study_export_raw_btn.setEnabled(True)
        self.study_clear_btn.setEnabled(True)
    
# 文件: FD-Advanced_3D.py

    def _display_aggregated_results(self):
        """显示聚合统计结果"""
        self.study_aggregated_results = self._aggregate_study_results()
        
        total_raw = len(self.study_results)
        total_groups = len(self.study_aggregated_results)
        status_text = self.tr("原始数据 {0} 条，聚合为 {1} 组参数组合").format(total_raw, total_groups)
        self.study_stats_label.setText(status_text)
        
        # --- 新增：定义新的、更详细的列标题 ---
        columns = [
            self.tr('立方体边长'), self.tr('理论D'), self.tr('理论N0'), 
            self.tr('实际3D D'), self.tr('实际3D N0'), 
            self.tr('总平均2D D'), self.tr('总平均2D N0'),
            self.tr('YOZ平均2D D'), self.tr('YOZ平均2D N0'),
            self.tr('XOZ平均2D D'), self.tr('XOZ平均2D N0'),
            self.tr('XOY平均2D D'), self.tr('XOY平均2D N0'),
            self.tr('展开-立方体D'), self.tr('展开-立方体N0'),
            self.tr('展开-圆柱体D'), self.tr('展开-圆柱体N0'),

            self.tr('重复次数')
        ]
        # 判断是否显示列：这里更严谨的做法是检查数据本身
        # 如果聚合结果中 slice_profile_dim_mean 不全为 None，则显示
        has_profile_data = False
        if self.study_aggregated_results:
            # 检查第一条数据是否有内容 (假设同批次一致)
            if self.study_aggregated_results[0].get('slice_profile_dim_mean') is not None:
                has_profile_data = True
        
        if has_profile_data:
            columns.append(self.tr('位置趋势分析'))
            
        columns.append(self.tr('重复次数'))
        
        self.study_results_table.setColumnCount(len(columns))
        self.study_results_table.setHorizontalHeaderLabels(columns)
        self.study_results_table.setRowCount(len(self.study_aggregated_results))
        
        for row, result in enumerate(self.study_aggregated_results):
            col = 0
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result['cube_size']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result['theoretical_dim']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result['theoretical_n0']))); col+=1
            
            def format_mean_std(mean, std):
                if mean is None: return "N/A"
                if std is None or std == 0: return f"{mean:.4f}"
                return f"{mean:.4f}±{std:.4f}"
            
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['actual_3d_dim_mean'], result['actual_3d_dim_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['actual_3d_n0_mean'], result['actual_3d_n0_std']))); col+=1
            
            # --- 新增：填充所有新2D数据到表格 ---
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['average_2d_dim_all_mean'], result['average_2d_dim_all_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['average_2d_n0_all_mean'], result['average_2d_n0_all_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['average_2d_dim_yoz_mean'], result['average_2d_dim_yoz_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['average_2d_n0_yoz_mean'], result['average_2d_n0_yoz_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['average_2d_dim_xoz_mean'], result['average_2d_dim_xoz_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['average_2d_n0_xoz_mean'], result['average_2d_n0_xoz_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['average_2d_dim_xoy_mean'], result['average_2d_dim_xoy_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['average_2d_n0_xoy_mean'], result['average_2d_n0_xoy_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['cuboid_unfolding_dim_mean'], result['cuboid_unfolding_dim_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['cuboid_unfolding_n0_mean'], result['cuboid_unfolding_n0_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['cylinder_unfolding_dim_mean'], result['cylinder_unfolding_dim_std']))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(format_mean_std(result['cylinder_unfolding_n0_mean'], result['cylinder_unfolding_n0_std']))); col+=1
            if has_profile_data:
                self._add_trend_button(row, col, result.get('slice_profile_dim_mean'), result.get('slice_profile_n0_mean'))
                col += 1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result['repeat_count']))); col+=1
        
        self.study_results_table.resizeColumnsToContents()
    
    def _display_raw_results(self):
        """显示原始详细结果（动态显示位置趋势列）"""
        status_text = self.tr("总计 {0} 条原始数据").format(len(self.study_results))
        self.study_stats_label.setText(status_text)
        
        # 1. 基础列定义
        columns = [
            self.tr('立方体边长'), self.tr('理论D'), self.tr('理论N0'), 
            self.tr('实际3D D'), self.tr('实际3D N0'), 
            self.tr('总平均2D D'), self.tr('总平均2D N0'),
            self.tr('YOZ平均2D D'), self.tr('YOZ平均2D N0'),
            self.tr('XOZ平均2D D'), self.tr('XOZ平均2D N0'),
            self.tr('XOY平均2D D'), self.tr('XOY平均2D N0'),
            self.tr('展开-立方体D'), self.tr('展开-立方体N0'),
            self.tr('展开-圆柱体D'), self.tr('展开-圆柱体N0')
        ]
        
        # 2. 动态判断是否需要显示“位置趋势分析”列
        # 检查第一条数据是否有该字段的内容
        has_profile_data = False
        if self.study_results:
            # 注意：原始数据中该字段可能是 None (未勾选计算)，也可能是 List 或 JSON String
            if self.study_results[0].get('slice_profile_dim') is not None:
                has_profile_data = True
        
        if has_profile_data:
            columns.append(self.tr('位置趋势分析'))
            
        columns.append(self.tr('重复序号'))

        # 3. 设置表头
        self.study_results_table.setColumnCount(len(columns))
        self.study_results_table.setHorizontalHeaderLabels(columns)
        self.study_results_table.setRowCount(len(self.study_results))
        
        # 4. 填充数据
        for row, result in enumerate(self.study_results):
            col = 0
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('cube_size', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('theoretical_dim', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('theoretical_n0', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('actual_3d_dim', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('actual_3d_n0', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('average_2d_dim_all', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('average_2d_n0_all', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('average_2d_dim_yoz', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('average_2d_n0_yoz', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('average_2d_dim_xoz', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('average_2d_n0_xoz', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('average_2d_dim_xoy', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('average_2d_n0_xoy', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('cuboid_unfolding_dim', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('cuboid_unfolding_n0', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('cylinder_unfolding_dim', 'N/A')))); col+=1
            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('cylinder_unfolding_n0', 'N/A')))); col+=1
            
            # === 仅当有数据时才添加按钮列 ===
            if has_profile_data:
                p_dim = result.get('slice_profile_dim')
                p_n0 = result.get('slice_profile_n0')
                
                # 兼容性处理：如果是字符串则解析，如果是列表则直接使用
                if isinstance(p_dim, str): p_dim = json.loads(p_dim)
                if isinstance(p_n0, str): p_n0 = json.loads(p_n0)
                
                self._add_trend_button(row, col, p_dim, p_n0)
                col += 1
            # =================================

            self.study_results_table.setItem(row, col, QTableWidgetItem(str(result.get('repeat_index', 'N/A')))); col+=1

        self.study_results_table.resizeColumnsToContents()
    
    def clear_2d3d_results(self):
        """清除2D-3D研究结果"""
        self.study_results = []
        self.study_aggregated_results = []
        self.current_session_timestamp = None  # <<< 新增：清除时重置时间戳
        self.study_results_table.setRowCount(0)
        self.study_stats_label.setText("暂无数据")
        self.study_copy_btn.setEnabled(True)  # 复制按钮始终可用
        self.study_save_btn.setEnabled(False)
        self.study_export_raw_btn.setEnabled(False)
        self.study_clear_btn.setEnabled(False)
        self.study_status_label.setText("请设置参数后点击开始计算")
    
# 文件: FD-Advanced_3D.py

# 文件: FD-Advanced_3D.py (替换 copy_2d3d_results)

    def copy_2d3d_results(self):
        """复制2D-3D研究结果到剪贴板"""
        if not self.study_results:
            return
        
        data = []
        current_index = self.study_display_mode_combo.currentIndex()
        
        if current_index == 0: # 索引0代表“统计汇总”
            # =================== 模式一：复制聚合统计数据 ===================
            columns = [
                self.tr('立方体边长'), self.tr('理论D'), self.tr('理论N0'), 
                self.tr('实际3D D'), self.tr('实际3D N0'), 
                self.tr('总平均2D D'), self.tr('总平均2D N0'),
                self.tr('YOZ平均2D D'), self.tr('YOZ平均2D N0'),
                self.tr('XOZ平均2D D'), self.tr('XOZ平均2D N0'),
                self.tr('XOY平均2D D'), self.tr('XOY平均2D N0'),
                self.tr('展开-立方体D'), self.tr('展开-立方体N0'),
                self.tr('展开-圆柱体D'), self.tr('展开-圆柱体N0'),
                self.tr('重复次数')
            ]
            data.append('\t'.join(columns))
            
            for result in self.study_aggregated_results:
                def format_mean_std(mean, std):
                    if mean is None: return "N/A"
                    if std is None or std == 0: return f"{mean:.4f}"
                    return f"{mean:.4f}±{std:.4f}"
                
                row_data = [
                    str(result['cube_size']), str(result['theoretical_dim']), str(result['theoretical_n0']),
                    format_mean_std(result['actual_3d_dim_mean'], result['actual_3d_dim_std']),
                    format_mean_std(result['actual_3d_n0_mean'], result['actual_3d_n0_std']),
                    format_mean_std(result['average_2d_dim_all_mean'], result['average_2d_dim_all_std']),
                    format_mean_std(result['average_2d_n0_all_mean'], result['average_2d_n0_all_std']),
                    format_mean_std(result['average_2d_dim_yoz_mean'], result['average_2d_dim_yoz_std']),
                    format_mean_std(result['average_2d_n0_yoz_mean'], result['average_2d_n0_yoz_std']),
                    format_mean_std(result['average_2d_dim_xoz_mean'], result['average_2d_dim_xoz_std']),
                    format_mean_std(result['average_2d_n0_xoz_mean'], result['average_2d_n0_xoz_std']),
                    format_mean_std(result['average_2d_dim_xoy_mean'], result['average_2d_dim_xoy_std']),
                    format_mean_std(result['average_2d_n0_xoy_mean'], result['average_2d_n0_xoy_std']),
                    format_mean_std(result['cuboid_unfolding_dim_mean'], result['cuboid_unfolding_dim_std']),
                    format_mean_std(result['cuboid_unfolding_n0_mean'], result['cuboid_unfolding_n0_std']),
                    format_mean_std(result['cylinder_unfolding_dim_mean'], result['cylinder_unfolding_dim_std']),
                    format_mean_std(result['cylinder_unfolding_n0_mean'], result['cylinder_unfolding_n0_std']),
                    str(result['repeat_count'])
                ]
                data.append('\t'.join(row_data))

            mode_text_for_msgbox = self.tr("统计汇总")
        else: # 索引1 (或其他) 代表“原始数据”
            # =================== 模式二：复制原始数据 ===================
            columns = [
                self.tr('立方体边长'), self.tr('理论D'), self.tr('理论N0'), 
                self.tr('实际3D D'), self.tr('实际3D N0'), 
                self.tr('总平均2D D'), self.tr('总平均2D N0'),
                self.tr('YOZ平均2D D'), self.tr('YOZ平均2D N0'),
                self.tr('XOZ平均2D D'), self.tr('XOZ平均2D N0'),
                self.tr('XOY平均2D D'), self.tr('XOY平均2D N0'),
                self.tr('展开-立方体D'), self.tr('展开-立方体N0'),
                self.tr('展开-圆柱体D'), self.tr('展开-圆柱体N0'),
                self.tr('重复序号')
            ]
            data.append('\t'.join(columns))

            for result in self.study_results:
                row_data = [
                    str(result.get('cube_size', 'N/A')),
                    str(result.get('theoretical_dim', 'N/A')),
                    str(result.get('theoretical_n0', 'N/A')),
                    str(result.get('actual_3d_dim', 'N/A')),
                    str(result.get('actual_3d_n0', 'N/A')),
                    str(result.get('average_2d_dim_all', 'N/A')),
                    str(result.get('average_2d_n0_all', 'N/A')),
                    str(result.get('average_2d_dim_yoz', 'N/A')),
                    str(result.get('average_2d_n0_yoz', 'N/A')),
                    str(result.get('average_2d_dim_xoz', 'N/A')),
                    str(result.get('average_2d_n0_xoz', 'N/A')),
                    str(result.get('average_2d_dim_xoy', 'N/A')),
                    str(result.get('average_2d_n0_xoy', 'N/A')),
                    str(result.get('cuboid_unfolding_dim', 'N/A')),
                    str(result.get('cuboid_unfolding_n0', 'N/A')),
                    str(result.get('cylinder_unfolding_dim', 'N/A')),
                    str(result.get('cylinder_unfolding_n0', 'N/A')),
                    str(result.get('repeat_index', 'N/A'))
                ]
                data.append('\t'.join(row_data))
            
            mode_text_for_msgbox = self.tr("原始数据")

        clipboard_text = '\n'.join(data)
        QApplication.clipboard().setText(clipboard_text)
        
        QMessageBox.information(self, self.tr("复制成功"), self.tr("{0}表格内容已复制到剪贴板").format(mode_text_for_msgbox))
    
# 文件: FD-Advanced_3D.py

# 文件: FD-Advanced_3D.py (替换 save_2d3d_results)

    def save_2d3d_results(self):
        """保存2D-3D研究结果为CSV文件"""
        if not self.study_results:
            return
        
        from datetime import datetime
        
        def build_parameter_filename():
            """内部函数，用于构建标准的文件名"""
            l_input = self.study_cube_lengths_input.text().strip()
            d_input = self.study_fractal_dims_input.text().strip()
            n0_input = self.study_n0_values_input.text().strip()
            timestamp = self.current_session_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"aggregated_L={l_input}_D={d_input}_N0={n0_input}_{timestamp}.csv"
        
        current_index = self.study_display_mode_combo.currentIndex()
        if current_index == 0: # 统计汇总模式
             default_name = build_parameter_filename()
             dialog_title = "保存统计汇总结果"
        else:
             self.export_raw_2d3d_results()
             return

        filename, _ = QFileDialog.getSaveFileName(
            self, self.tr(dialog_title), default_name, "CSV文件 (*.csv)"
        )
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    
                    columns = [
                        '立方体边长', '理论D', '理论N0', 
                        '实际3D_D_平均值', '实际3D_D_标准差', '实际3D_N0_平均值', '实际3D_N0_标准差',
                        '总平均2D_D_平均值', '总平均2D_D_标准差', '总平均2D_N0_平均值', '总平均2D_N0_标准差',
                        'YOZ平均2D_D_平均值', 'YOZ平均2D_D_标准差', 'YOZ平均2D_N0_平均值', 'YOZ平均2D_N0_标准差',
                        'XOZ平均2D_D_平均值', 'XOZ平均2D_D_标准差', 'XOZ平均2D_N0_平均值', 'XOZ平均2D_N0_标准差',
                        'XOY平均2D_D_平均值', 'XOY平均2D_D_标准差', 'XOY平均2D_N0_平均值', 'XOY平均2D_N0_标准差',
                        '展开-立方体D_平均值', '展开-立方体D_标准差', '展开-立方体N0_平均值', '展开-立方体N0_标准差',
                        '展开-圆柱体D_平均值', '展开-圆柱体D_标准差', '展开-圆柱体N0_平均值', '展开-圆柱体N0_标准差',
                        '重复次数'
                    ]
                    writer.writerow(columns)
                    
                    for result in self.study_aggregated_results:
                        writer.writerow([
                            result['cube_size'], result['theoretical_dim'], result['theoretical_n0'],
                            result.get('actual_3d_dim_mean'), result.get('actual_3d_dim_std'),
                            result.get('actual_3d_n0_mean'), result.get('actual_3d_n0_std'),
                            result.get('average_2d_dim_all_mean'), result.get('average_2d_dim_all_std'),
                            result.get('average_2d_n0_all_mean'), result.get('average_2d_n0_all_std'),
                            result.get('average_2d_dim_yoz_mean'), result.get('average_2d_dim_yoz_std'),
                            result.get('average_2d_n0_yoz_mean'), result.get('average_2d_n0_yoz_std'),
                            result.get('average_2d_dim_xoz_mean'), result.get('average_2d_dim_xoz_std'),
                            result.get('average_2d_n0_xoz_mean'), result.get('average_2d_n0_xoz_std'),
                            result.get('average_2d_dim_xoy_mean'), result.get('average_2d_dim_xoy_std'),
                            result.get('average_2d_n0_xoy_mean'), result.get('average_2d_n0_xoy_std'),
                            result.get('cuboid_unfolding_dim_mean'), result.get('cuboid_unfolding_dim_std'),
                            result.get('cuboid_unfolding_n0_mean'), result.get('cuboid_unfolding_n0_std'),
                            result.get('cylinder_unfolding_dim_mean'), result.get('cylinder_unfolding_dim_std'),
                            result.get('cylinder_unfolding_n0_mean'), result.get('cylinder_unfolding_n0_std'),
                            result['repeat_count']
                        ])
                
                QMessageBox.information(self, self.tr("保存成功"), self.tr("统计汇总结果已保存到:\n{0}").format(filename))
                
            except Exception as e:
                QMessageBox.critical(self, self.tr("保存失败"), self.tr("保存失败:\n{0}").format(str(e)))
    
# 文件: FD-Advanced_3D.py

# 文件: FD-Advanced_3D.py (替换 export_raw_2d3d_results)

    def export_raw_2d3d_results(self):
        """导出原始数据为CSV文件"""
        if not self.study_results:
            return
        
        from datetime import datetime
        
        def build_parameter_filename():
            l_input = self.study_cube_lengths_input.text().strip()
            d_input = self.study_fractal_dims_input.text().strip()
            n0_input = self.study_n0_values_input.text().strip()
            timestamp = self.current_session_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"L={l_input}_D={d_input}_N0={n0_input}_raw_{timestamp}.csv"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, self.tr("导出原始数据"), 
            build_parameter_filename(),
            "CSV文件 (*.csv)"
        )
        
        if filename:
            try:
                import csv
                columns = [
                    '立方体边长', '理论D', '理论N0', 
                    '实际3D_D', '实际3D_N0', 
                    '总平均2D_D', '总平均2D_N0',
                    'YOZ平均2D_D', 'YOZ平均2D_N0',
                    'XOZ平均2D_D', 'XOZ平均2D_N0',
                    'XOY平均2D_D', 'XOY平均2D_N0',
                    '展开-立方体D', '展开-立方体N0',
                    '展开-圆柱体D', '展开-圆柱体N0',
                    '重复序号'
                ]
                
                with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    
                    for result in self.study_results:
                        writer.writerow([
                            result.get('cube_size'),
                            result.get('theoretical_dim'),
                            result.get('theoretical_n0'),
                            result.get('actual_3d_dim'),
                            result.get('actual_3d_n0'),
                            result.get('average_2d_dim_all'),
                            result.get('average_2d_n0_all'),
                            result.get('average_2d_dim_yoz'),
                            result.get('average_2d_n0_yoz'),
                            result.get('average_2d_dim_xoz'),
                            result.get('average_2d_n0_xoz'),
                            result.get('average_2d_dim_xoy'),
                            result.get('average_2d_n0_xoy'),
                            result.get('cuboid_unfolding_dim'),
                            result.get('cuboid_unfolding_n0'),
                            result.get('cylinder_unfolding_dim'),
                            result.get('cylinder_unfolding_n0'),
                            result.get('repeat_index')
                        ])
                
                QMessageBox.information(self, self.tr("导出成功"), self.tr(f"原始数据已导出到:\n{filename}"))
                
            except Exception as e:
                QMessageBox.critical(self, self.tr("导出失败"), self.tr(f"导出失败:\n{str(e)}"))
    
# 文件: FD-Advanced_3D.py

# 文件: FD-Advanced_3D.py (替换 load_backup_file)

    def load_backup_file(self):
        """加载SQLite备份文件并驱动恢复流程（重构版）"""
        import sqlite3
        
        filename, _ = QFileDialog.getOpenFileName(
            self, self.tr("选择备份文件"), "", "SQLite数据库 (*.db);;所有文件 (*.*)"
        )
        if not filename:
            return

        # --- 新增：从加载的文件名中提取时间戳 ---
        basename = os.path.basename(filename)
        match = re.search(r"fractal_calculation_backup_(\d{8}_\d{6})\.db", basename)
        if match:
            self.current_session_timestamp = match.group(1)
        else:
            self.current_session_timestamp = None
            QMessageBox.warning(self, self.tr("警告"), 
                                self.tr("无法从文件名 '{0}' 中提取时间戳。\n"
                                        "若保存CSV，将使用当前时间作为文件名。").format(basename))
        # --- 新增结束 ---

        try:
            backup_manager = SQLiteBackupManager(filename)
            backup_data = backup_manager.load_backup_data()
            
            if not backup_data or 'results' not in backup_data:
                QMessageBox.warning(self, self.tr("加载失败"), self.tr("备份文件中没有找到有效的结果数据。"))
                return

            results_data = backup_data.get('results', [])
            metadata = backup_data.get('metadata', {})
            calculation_params = metadata.get('calculation_params', {})
            
            # 关键验证：检查文件是否包含可用于恢复的参数
            can_continue = calculation_params and calculation_params.get('gui_input_params')

            # 无论如何，先加载结果数据
            self.study_results = results_data
            self.update_2d3d_results_display()
            self.study_save_btn.setEnabled(True)
            self.study_export_raw_btn.setEnabled(True)
            self.study_clear_btn.setEnabled(True)
            self.study_start_calc_btn.setEnabled(True)
            self.current_backup_file = filename
            
            if not can_continue:
                # 场景1：文件信息不全，只加载数据并提示
                self.study_status_label.setText(self.tr("已加载 {0} 条结果。").format(len(results_data)))
                QMessageBox.information(self, 
                    self.tr("仅加载数据"),
                    self.tr("结果已成功加载并显示。\n\n"
                            "注意：此备份文件缺少恢复计算所需的参数信息，无法继续计算。"))
                return

            # --- 从这里开始，是文件信息完整，可以恢复的流程 ---
            
            # 步骤2：填充UI并分析进度
            self._fill_gui_from_params(calculation_params)
            completed_tasks = self._analyze_completed_tasks(backup_data, calculation_params)
            
            # 步骤3：计算进度
            total_combinations = (len(calculation_params.get('cube_lengths', [])) *
                                  len(calculation_params.get('fractal_dims', [])) *
                                  len(calculation_params.get('n0_values', [])))
            repeat_count = calculation_params.get('repeat_count', 1)
            total_simulations = total_combinations * repeat_count
            completed_count = len(results_data) # 直接用结果数量更准确
            remaining_count = total_simulations - completed_count
            
            self.study_status_label.setText(
                self.tr("已加载 {0} / {1} 条结果。").format(completed_count, total_simulations)
            )

            # 步骤4：决策与交互
            if remaining_count <= 0:
                QMessageBox.information(self, self.tr("任务已完成"), self.tr("此计算任务已全部完成，无需继续。"))
                return
            
            reply = QMessageBox.question(
                self,
                self.tr("继续计算？"),
                self.tr("检测到未完成的计算任务。\n\n"
                        "总任务: {0}\n"
                        "已完成: {1}\n"
                        "剩余: {2}\n\n"
                        "是否要继续计算剩余的任务？").format(total_simulations, completed_count, remaining_count),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            # 步骤5：执行
            if reply == QMessageBox.Yes:
                self._start_continue_calculation(calculation_params, completed_tasks)
            
        except sqlite3.Error as e:
            QMessageBox.critical(self, self.tr("数据库错误"), self.tr("SQLite数据库错误:\n{0}").format(str(e)))
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, self.tr("加载失败"), self.tr("加载备份文件时发生未知错误:\n{0}").format(error_msg))

    def _continue_calculation_from_backup(self, backup_data):
        """从备份文件继续计算（重构优化版，不再提问）"""
        try:
            calculation_params = backup_data.get('calculation_params', {})
            results_data = backup_data.get('results', [])
            
            # 1. 填充GUI参数
            self._fill_gui_from_params(calculation_params)
            
            # 2. 加载已有结果并更新显示
            self.study_results = results_data
            self.update_2d3d_results_display()
            
            # 3. 分析已完成的任务
            # 注意：这里传递的是整个 backup_data 字典
            completed_tasks = self._analyze_completed_tasks(backup_data, calculation_params)
            
            # 4. 启动继续计算的流程
            self._start_continue_calculation(calculation_params, completed_tasks)
                
        except Exception as e:
            QMessageBox.critical(self, self.tr("继续计算失败"), self.tr("从备份文件准备继续计算时发生错误:\n{0}").format(str(e)))
    
# --- START OF FILE FD-Advanced_3D.py (in class FractalFractureGeneratorGUI) ---

    def _fill_gui_from_params(self, params):
        """从参数字典填充GUI控件（增强版，包含裂缝形态参数）"""
        # 1. 恢复 2D-3D 研究标签页的参数
        gui_params = params.get('gui_input_params', {})
        self.study_cube_lengths_input.setText(gui_params.get('cube_lengths_str', ''))
        self.study_fractal_dims_input.setText(gui_params.get('fractal_dims_str', ''))
        self.study_n0_values_input.setText(gui_params.get('n0_values_str', ''))
        self.study_repeat_count_spin.setValue(params.get('repeat_count', 1))
        self.study_gen_iterations_spin.setValue(params.get('gen_iterations', 3))
        self.study_calc_iterations_spin.setValue(params.get('calc_iterations', 3))
        self.study_num_slices_spin.setValue(params.get('num_slices', 21)) # 使用顶层参数

        # 优先从 gui_input_params 获取，如果没有则从顶层 params 获取，默认 False
        gui_params = params.get('gui_input_params', {})
        profile_checked = gui_params.get('calc_slice_profile', params.get('calc_slice_profile', False))
        self.study_calc_profile_checkbox.setChecked(profile_checked)

        center_only_checked = gui_params.get('calc_center_only', params.get('calc_center_only', False))
        self.study_center_only_checkbox.setChecked(center_only_checked)

        self.study_enable_parallel_checkbox.setChecked(params.get('max_processes', 1) > 1)
        self.study_thread_count_spin.setValue(params.get('max_processes', 1))

        # 2. 恢复 主界面 的裂缝形态参数
        # 使用 .get(key, default_value) 来安全地获取参数，兼容旧的备份文件
        
        # 2a. 恢复长短轴比例参数
        random_aspect = params.get('random_aspect_ratio', False)
        self.random_aspect_ratio_check.setChecked(random_aspect)
        self.aspect_ratio_base_spin.setValue(params.get('aspect_ratio_base', 1.0))
        self.aspect_ratio_variation_spin.setValue(params.get('aspect_ratio_variation', 0.0))
        # 触发UI更新以显示/隐藏控件
        self.toggle_aspect_ratio_controls(random_aspect)

        # 2b. 恢复产状分布参数
        is_isotropic = params.get('is_isotropic', True)
        self.isotropic_check.setChecked(is_isotropic)
        self.mean_inclination_spin.setValue(params.get('mean_inclination', 90.0))
        self.mean_azimuth_spin.setValue(params.get('mean_azimuth', 0.0))
        self.kappa_spin.setValue(params.get('kappa', 20.0))
        # 触发UI更新以启用/禁用控件
        self.toggle_orientation_mode(is_isotropic)

# --- END OF REPLACEMENT for _fill_gui_from_params ---
    
    def _analyze_completed_tasks(self, results_data, calculation_params):
        """分析已完成的任务（与_generate_task_queue逻辑匹配的最终版）"""
        completed_tasks_by_key = []

        results_list = []
        if isinstance(results_data, dict) and 'results' in results_data:
            results_list = results_data['results']
        elif isinstance(results_data, list):
            results_list = results_data
        
        for result in results_list:
            # 创建一个与 _generate_task_queue 中 task_key 完全一致的元组
            task_key = (
                result.get('cube_size'),
                result.get('theoretical_dim'),
                result.get('theoretical_n0')
            )
            completed_tasks_by_key.append(task_key)
            
        return completed_tasks_by_key
    


    def _start_continue_calculation(self, calculation_params, completed_tasks):
        """
        启动一个“继续计算”的任务（最终修复版，修正并行判断逻辑）。
        """
        try:
            # 从参数字典中安全地获取所有需要的数据
            cube_lengths = calculation_params.get('cube_lengths', [])
            fractal_dims = calculation_params.get('fractal_dims', [])
            n0_values = calculation_params.get('n0_values', [])
            repeat_count = calculation_params.get('repeat_count', 1)
            gui_params = calculation_params.get('gui_input_params', {})
            gen_iterations = calculation_params.get('gen_iterations', 3)
            calc_iterations = calculation_params.get('calc_iterations', 3)
            num_slices = calculation_params.get('num_slices', 21)

            # === 新增：获取开关参数 ===
            calc_slice_profile = calculation_params.get('calc_slice_profile', False)
            calc_center_only = calculation_params.get('calc_center_only', False) # 现在这两个变量将在下面被使用，变亮了

            total_simulations = len(cube_lengths) * len(fractal_dims) * len(n0_values) * repeat_count
            completed_count = len(self.study_results)

            # 获取椭圆形态参数
            # 形态参数 (长短轴)
            random_aspect_ratio = calculation_params.get('random_aspect_ratio', False)
            aspect_ratio_base = calculation_params.get('aspect_ratio_base', 1.0)
            aspect_ratio_variation = calculation_params.get('aspect_ratio_variation', 0.0)

            # 形态参数 (产状/Fisher)
            is_isotropic = calculation_params.get('is_isotropic', True)
            mean_inclination = calculation_params.get('mean_inclination', 90.0)
            mean_azimuth = calculation_params.get('mean_azimuth', 0.0)
            kappa = calculation_params.get('kappa', 0.0)

            
            # --- 核心修复：直接从顶层参数读取进程数来判断并行模式 ---
            # 'max_processes' 是在线程 __init__ 中直接存入 calculation_params 的，来源更可靠。
            # 如果是单线程任务，这个键不存在，get() 会返回默认值 1。
            thread_count = calculation_params.get('max_processes', 1)
            enable_parallel = thread_count > 1
            # --- 修复结束 ---
            
            # 初始化计算界面
            self.study_start_calc_btn.setEnabled(False)
            self.study_stop_calc_btn.setEnabled(True)
            self.study_progress_bar.setVisible(True)
            self.study_progress_bar.setRange(0, total_simulations)
            self.study_progress_bar.setValue(completed_count)
            self.completed_tasks_offset = completed_count
            
            # 启动计算线程
            if enable_parallel:
                print(f"继续计算：检测到并行模式，将使用 {thread_count} 个进程。")
                self.study_2d3d_thread = Study2D3DParallelCalculationThread(
                    # 1. 基础分形参数
                    cube_lengths, fractal_dims, n0_values, repeat_count,
                    gen_iterations, calc_iterations,
                    num_slices, 
                    # === 修复：在这里插入刚才获取的参数 ===
                    calc_slice_profile,     # 8
                    calc_center_only,       # 9
                    # ===================================
                    # 2. 形态参数
                    random_aspect_ratio, aspect_ratio_base, aspect_ratio_variation,
                    is_isotropic, mean_inclination, mean_azimuth, kappa,
                    # 3. 必须按位置传递的 results_queue
                    self.results_queue,     # 16
                    # 4. 之后的所有参数都使用关键字形式传递
                    max_processes=thread_count,
                    backup_file_path=self.current_backup_file,
                    gui_input_params=gui_params,
                    completed_tasks=completed_tasks
                )
            else:
                print("继续计算：检测到单线程模式。")
                self.study_2d3d_thread = Study2D3DCalculationThread(
                    # 1. 基础分形参数
                    cube_lengths, fractal_dims, n0_values, repeat_count,
                    gen_iterations, calc_iterations,
                    num_slices,
                    # === 修复：在这里插入刚才获取的参数 ===
                    calc_slice_profile,     # 8
                    calc_center_only,       # 9
                    # ===================================
                    # 2. 形态参数
                    random_aspect_ratio, aspect_ratio_base, aspect_ratio_variation,
                    is_isotropic, mean_inclination, mean_azimuth, kappa,
                    # 3. 关键字参数
                    completed_tasks=completed_tasks,
                    backup_file_path=self.current_backup_file,
                    gui_input_params=gui_params
                )
            
            # --- (信号连接部分，这是修改点) ---
            self.study_2d3d_thread.calculation_finished.connect(self.on_study_calculation_finished)
            self.study_2d3d_thread.calculation_failed.connect(self.on_study_calculation_failed)
            
            # 关键修改：单线程模式现在也使用缓冲区，不再直接连接UI更新函数
            if not enable_parallel:
                self.study_2d3d_thread.single_result_ready.connect(self.result_buffer.append)
            
            self.study_2d3d_thread.start()

            # <<< 新增：在这里启动UI更新定时器 >>>
            self.ui_update_timer.start()

            mode_text = self.tr("并行模式({0}进程)").format(thread_count) if enable_parallel else self.tr("单线程模式")
            status_text = self.tr("继续计算中... ({0})\n备份文件: {1}").format(mode_text, self.current_backup_file)
            self.study_status_label.setText(status_text)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, self.tr("继续计算失败"), self.tr("开始继续计算时发生错误:\n{0}").format(error_msg))
            self.study_start_calc_btn.setEnabled(True)
            self.study_stop_calc_btn.setEnabled(False)

    def capture_current_3d_view(self):
        """捕获并保存当前标签页中显示的3D视图。"""
        if not self.tab_widget or self.tab_widget.count() == 0:
            QMessageBox.warning(self, self.tr("警告"), self.tr("没有可供捕获的视图。"))
            return

        # 获取当前活动的标签页控件
        current_widget = self.tab_widget.currentWidget()

        # 检查它是否是一个有效的Matplotlib画布
        if isinstance(current_widget, MplCanvas):
            # 通过检查是否有z轴来确认它是一个3D图
            if hasattr(current_widget.axes, 'get_zlim'):
                # 调用通用的图表保存函数
                self._save_figure_to_file(self, current_widget.fig, "3d_view_capture")
            else:
                QMessageBox.warning(self, self.tr("操作无效"), self.tr("请选择一个包含三维裂缝网络视图的标签页进行捕获。"))
        else:
            QMessageBox.warning(self, self.tr("操作无效"), self.tr("当前标签页不是一个有效的图表视图。"))

    def plot_slice_trend(self):
        """绘制切片位置与分形参数的趋势关系图（支持三向平均）"""
        if not self.latest_face_analysis_data:
            QMessageBox.warning(self, self.tr("警告"), self.tr("没有切片分析数据，请先生成模型并确保完成了六面分析计算。"))
            return

        current_index = self.section_type_combo.currentIndex()
        
        # 准备数据容器
        positions = []
        d_values = []
        n0_values = []
        axis_label = ""
        
        # ==================== 逻辑分支 A: 单一方向 (原逻辑) ====================
        if current_index < 3:
            target_prefix = ""
            if current_index == 0:
                target_prefix = "x="; axis_label = "X"
            elif current_index == 1:
                target_prefix = "y="; axis_label = "Y"
            elif current_index == 2:
                target_prefix = "z="; axis_label = "Z"
            
            # 筛选并排序数据
            relevant_data = []
            for item in self.latest_face_analysis_data:
                if item['face_name'].startswith(target_prefix):
                    try:
                        pos = float(item['face_name'].split('=')[1])
                        relevant_data.append((pos, item))
                    except (IndexError, ValueError): continue
            
            relevant_data.sort(key=lambda x: x[0])
            
            for pos, item in relevant_data:
                if item['fractal_dim'] is not None:
                    positions.append(pos)
                    d_values.append(item['fractal_dim'])
                    n0_values.append(10 ** item['fit_data']['coeffs'][1])

        # ==================== 逻辑分支 B: 三个方向平均 (新逻辑) ====================
        else: # current_index == 3
            axis_label = self.tr("位置")
            
            # 1. 将数据按位置分组
            # 结构: position_map[0.500] = {'x': item, 'y': item, 'z': item}
            position_map = {}
            
            for item in self.latest_face_analysis_data:
                name = item['face_name'] # e.g., "x=0.500"
                if '=' not in name: continue
                
                axis, val_str = name.split('=')
                try:
                    pos = float(val_str)
                    # 使用字符串作为key避免浮点数精度问题导致的分组错误
                    pos_key = f"{pos:.3f}" 
                    
                    if pos_key not in position_map:
                        position_map[pos_key] = {'val': pos, 'data': []}
                    
                    # 只有当该面有有效的分形数据时才加入
                    if item['fractal_dim'] is not None:
                        position_map[pos_key]['data'].append(item)
                        
                except ValueError: continue

            # 2. 对每个位置进行聚合计算
            # 我们需要按位置排序
            sorted_keys = sorted(position_map.keys(), key=lambda k: position_map[k]['val'])
            
            for key in sorted_keys:
                entry = position_map[key]
                items = entry['data']
                pos = entry['val']
                
                # 如果这个位置没有数据，或数据不全（可选：是否必须三个面都有？这里假设只要有数据就计算）
                if not items: continue

                # --- 聚合统计值 ---
                # 我们需要合并 items 中所有 levels 的 valid_count
                # levels 结构: [{'box_size': 1.0, 'valid_count': 100}, ...]
                
                aggregated_levels = {} # box_size -> list of counts
                
                for item in items:
                    for level in item['levels']:
                        bs = level['box_size']
                        if bs not in aggregated_levels:
                            aggregated_levels[bs] = []
                        aggregated_levels[bs].append(level['valid_count'])
                
                # 计算每个 box_size 的平均 count
                avg_level_data = []
                for bs, counts in aggregated_levels.items():
                    if len(counts) > 0:
                        avg_count = sum(counts) / len(counts)
                        if avg_count > 0: # 过滤掉0值，因为要取对数
                            avg_level_data.append((bs, avg_count))
                
                # --- 重新拟合 (Regression) ---
                if len(avg_level_data) >= 2:
                    try:
                        box_sizes, counts = zip(*avg_level_data)
                        log_sizes = np.log10(box_sizes)
                        log_counts = np.log10(counts)
                        
                        # 线性拟合: y = kx + b (log_N = -D * log_L + log_N0)
                        coeffs = np.polyfit(log_sizes, log_counts, 1)
                        
                        new_d = -coeffs[0]
                        new_n0 = 10 ** coeffs[1]
                        
                        positions.append(pos)
                        d_values.append(new_d)
                        n0_values.append(new_n0)
                    except Exception:
                        continue # 拟合失败则跳过该点

        # ==================== 绘图 ====================
        if not positions:
            QMessageBox.information(self, self.tr("提示"), self.tr("未找到有效的切片分析数据用于绘制趋势图。"))
            return

        dialog = SliceTrendDialog(positions, d_values, n0_values, axis_label, self)
        # 如果是平均模式，修改一下标题前缀
        if current_index == 3:
            dialog.setWindowTitle(self.tr("切片位置分形特征趋势分析 (三向平均)"))
        dialog.exec_()

# 在 FractalFractureGeneratorGUI 类中添加
    def _add_trend_button(self, row, col, profile_dim, profile_n0):
        """在表格指定位置添加查看趋势的按钮"""
        if not profile_dim or not profile_n0:
            item = QTableWidgetItem(self.tr("无数据"))
            item.setFlags(Qt.ItemIsEnabled) # 禁用
            self.study_results_table.setItem(row, col, item)
            return

        btn = QPushButton(self.tr("查看曲线"))
        # 使用 lambda 闭包捕获数据
        btn.clicked.connect(lambda: self.show_slice_trend_from_table(profile_dim, profile_n0))
        self.study_results_table.setCellWidget(row, col, btn)

    def show_slice_trend_from_table(self, d_data, n0_data):
        """显示来自表格数据的趋势图"""
        # 生成归一化坐标 (0.0 ~ 1.0)
        count = len(d_data)
        if count == 0: return
        
        # 处理 None 值 (绘图时需要过滤或处理)
        x_data = []
        clean_d = []
        clean_n0 = []
        
        for i in range(count):
            if d_data[i] is not None and n0_data[i] is not None:
                x = i / (count - 1) if count > 1 else 0.5
                x_data.append(x)
                clean_d.append(d_data[i])
                clean_n0.append(n0_data[i])
        
        if not x_data:
            QMessageBox.information(self, self.tr("提示"), self.tr("该组数据中包含无效值，无法绘图。"))
            return

        # 复用之前定义的 SliceTrendDialog
        # 注意：这里 axis_label 使用 "相对位置"
        dialog = SliceTrendDialog(x_data, clean_d, clean_n0, self.tr("相对位置 (0-1)"), self)
        dialog.setWindowTitle(self.tr("分形参数位置效应分析 (多次模拟平均)"))
        dialog.exec_()
    # ==========================================================
    # 以下是为重构钻孔研究功能而新增或修改的函数
    # ==========================================================
# THIS IS THE NEW CODE
    def open_drilling_study_tab(self):
        """按需创建或切换到钻孔模拟研究标签页"""
        # --- 核心修改: 查找逻辑也应基于 objectName ---
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if widget and widget.objectName() == "drilling_study_tab_widget":
                self.tab_widget.setCurrentIndex(i)
                return

        study_widget = QWidget()
        
        # --- 核心修改: 使用 objectName 代替成员变量 ---
        study_widget.setObjectName("drilling_study_tab_widget")

        main_layout = QHBoxLayout()
        params_panel = self.create_drilling_params_panel()
        main_layout.addWidget(params_panel)
        results_area = self.create_drilling_results_area()
        main_layout.addWidget(results_area)
        main_layout.setStretch(0, 3)
        main_layout.setStretch(1, 7)
        study_widget.setLayout(main_layout)

        # 使用临时标题
        self.tab_widget.addTab(study_widget, "...")
        self.tab_widget.setCurrentWidget(study_widget)

        self.retranslate_ui()

    def create_drilling_params_panel(self):
        """创建钻孔研究标签页的左侧参数面板（功能简化版）"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMaximumWidth(500)

        layout = QVBoxLayout()

        self.drilling_title_label = QLabel() # 修改
        self.drilling_title_label.setAlignment(Qt.AlignCenter)
        self.drilling_title_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #2E86AB; margin: 10px;")
        layout.addWidget(self.drilling_title_label)
        
        self.drilling_display_options_group = QGroupBox() # 修改
        self.drilling_display_options_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        display_options_layout = QVBoxLayout()
        
        self.show_non_intersected_check = QCheckBox() # 修改
        self.show_non_intersected_check.setChecked(True)
        self.show_non_intersected_check.setStyleSheet("font-size: 14px;")
        self.show_non_intersected_check.stateChanged.connect(self.on_display_option_changed)
        display_options_layout.addWidget(self.show_non_intersected_check)
        
        opacity_form_layout = QFormLayout()
        
        self.fracture_opacity_spin = QDoubleSpinBox()
        self.fracture_opacity_spin.setRange(0.1, 1.0)
        self.fracture_opacity_spin.setValue(0.3)
        self.fracture_opacity_spin.setDecimals(2)
        self.fracture_opacity_spin.setSingleStep(0.1)
        self.fracture_opacity_spin.setStyleSheet("font-size: 14px;")
        self.fracture_opacity_label = QLabel() # 修改
        opacity_form_layout.addRow(self.fracture_opacity_label, self.fracture_opacity_spin)
        
        self.non_intersected_opacity_spin = QDoubleSpinBox()
        self.non_intersected_opacity_spin.setRange(0.01, 1.0)
        self.non_intersected_opacity_spin.setValue(0.05)
        self.non_intersected_opacity_spin.setDecimals(2)
        self.non_intersected_opacity_spin.setSingleStep(0.01)
        self.non_intersected_opacity_spin.setStyleSheet("font-size: 14px;")
        self.non_intersected_opacity_label = QLabel() # 修改
        opacity_form_layout.addRow(self.non_intersected_opacity_label, self.non_intersected_opacity_spin)
        
        display_options_layout.addLayout(opacity_form_layout)
        self.drilling_display_options_group.setLayout(display_options_layout)
        layout.addWidget(self.drilling_display_options_group)

        self.drilling_point_info_group = QGroupBox() # 修改
        self.drilling_point_info_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        point_info_layout = QVBoxLayout()
        self.random_point_label = QLabel() # 修改
        self.random_point_label.setWordWrap(True)
        self.random_point_label.setStyleSheet("color: #555; font-size: 14px;")
        point_info_layout.addWidget(self.random_point_label)
        self.drilling_point_info_group.setLayout(point_info_layout)
        layout.addWidget(self.drilling_point_info_group)
        
        self.drilling_result_info_group = QGroupBox() # 修改
        self.drilling_result_info_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        result_info_layout = QVBoxLayout()
        self.analysis_result_label = QLabel() # 修改
        self.analysis_result_label.setWordWrap(True)
        self.analysis_result_label.setStyleSheet("color: #555; font-size: 14px;")
        result_info_layout.addWidget(self.analysis_result_label)
        self.drilling_result_info_group.setLayout(result_info_layout)
        layout.addWidget(self.drilling_result_info_group)
        
        button_layout = QVBoxLayout()
        self.generate_random_point_btn = QPushButton() # 修改
        self.generate_random_point_btn.clicked.connect(self.generate_random_point_for_analysis)
        self.generate_random_point_btn.setEnabled(True if self.latest_fractures else False)
        self.generate_random_point_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.generate_random_point_btn)
        
        self.start_analysis_btn = QPushButton() # 修改
        self.start_analysis_btn.clicked.connect(self.start_drilling_analysis)
        self.start_analysis_btn.setEnabled(False)
        self.start_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.start_analysis_btn)
        
        self.clear_analysis_btn = QPushButton() # 修改
        self.clear_analysis_btn.clicked.connect(self.clear_drilling_analysis)
        self.clear_analysis_btn.setEnabled(False)
        self.clear_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.clear_analysis_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_drilling_results_area(self):
        """创建钻孔研究标签页的右侧结果显示区"""
        results_widget = QWidget()
        layout = QVBoxLayout()
        
        main_splitter = QSplitter(Qt.Vertical)
        
        self.drilling_probability_canvas = MplCanvas(self, width=12, height=8, dpi=100, is_3d=True)
        # 初始文本将在 retranslate_ui 中设置
        main_splitter.addWidget(self.drilling_probability_canvas)
        
        table_widget = QWidget()
        table_layout = QVBoxLayout()
        self.drilling_table_label = QLabel() # 修改
        self.drilling_table_label.setAlignment(Qt.AlignCenter)
        self.drilling_table_label.setStyleSheet("font-weight: bold; margin: 5px; font-size: 14px; color: #2E86AB;")
        table_layout.addWidget(self.drilling_table_label)
        
        self.drilling_table = QTableWidget()
        self.drilling_table.setColumnCount(9)
        # 表头将在 retranslate_ui 中设置
        header = self.drilling_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.drilling_table)
        table_widget.setLayout(table_layout)
        
        main_splitter.addWidget(table_widget)
        main_splitter.setSizes([700, 300])
        
        layout.addWidget(main_splitter)
        results_widget.setLayout(layout)
        return results_widget

    
    def create_display_area(self, parent):
        """创建显示区域"""
        # 创建标签页控件
        self.tab_widget = QTabWidget()
        parent.addWidget(self.tab_widget)
        
        # 初始创建默认标签页（会在生成时重新创建）
        self._create_default_tabs()
    
    def _create_default_tabs(self):
        """创建默认标签页"""
        self.canvases = {}
        
        # 清空现有标签页
        self.tab_widget.clear()
        
        # 创建3个默认标签页
        for i in range(3):
            canvas = MplCanvas(self, width=8, height=6, dpi=100, is_3d=True)
            self.canvases[i] = canvas
            
            if i == 0:
                tab_name = self.tr("初始状态")
            elif i == 1:
                tab_name = self.tr("第一次迭代")
            else:
                tab_name = self.tr("第二次迭代")
            
            self.tab_widget.addTab(canvas, tab_name)
        
        # 初始化画布（不调用clear_display避免递归）
        self._initialize_canvases()
    
    def _initialize_canvases(self):
        """初始化画布显示"""
        cube_size = self.cube_size_spin.value()
        
        for canvas in self.canvases.values():
            canvas.axes.clear()
            
            self.draw_cube_outline(canvas.axes, cube_size)
            
            canvas.axes.set_xlim([0, cube_size])
            canvas.axes.set_ylim([0, cube_size])
            canvas.axes.set_zlim([0, cube_size])
            
            # --- 核心修改 ---
            canvas.axes.set_xlabel(self.tr('X轴'), fontfamily='Microsoft YaHei')
            canvas.axes.set_ylabel(self.tr('Y轴'), fontfamily='Microsoft YaHei')
            canvas.axes.set_zlabel(self.tr('Z轴'), fontfamily='Microsoft YaHei')
            canvas.axes.set_title(self.tr('空间裂缝分布'), fontfamily='Microsoft YaHei')
            # --- 修改结束 ---
            
            canvas.axes.set_box_aspect([1,1,1])
            
            canvas.draw()
    
# 文件: FD-Advanced_3D.py

# 文件: FD-Advanced_3D.py
# 请找到 _create_dynamic_tabs 函数并用下面的代码块完整替换它

    def _create_dynamic_tabs(self, iterations):
        """根据迭代次数动态创建标签页"""
        self.canvases = {}
        
        # 清空现有标签页
        self.tab_widget.clear()
        
        # 创建迭代次数+1个标签页
        for i in range(iterations + 1):
            canvas = MplCanvas(self, width=8, height=6, dpi=100, is_3d=True)
            
            # 为每个迭代画布设置唯一的对象名称，以便 retranslate_ui 识别
            canvas.setObjectName(f"iteration_tab_canvas_{i}")
            
            self.canvases[i] = canvas
            
            # 添加标签页，标题是一个临时占位符，马上会被 retranslate_ui 覆盖
            self.tab_widget.addTab(canvas, f"...") 
        
        # 添加统计分析标签页
        self._create_statistics_analysis_tab()
        
        # 添加六个面投影轨迹分析标签页
        self._create_projection_analysis_tab()

        # --- 核心修复 ---
        # 在所有标签页被重新创建之后，立即调用 retranslate_ui
        # 来应用当前语言的正确翻译。
        self.retranslate_ui()
        # --- 修复结束 ---
        

    

        
    def _create_projection_analysis_tab(self):
        """创建六个面投影轨迹分析标签页"""
        from PyQt5.QtWidgets import QScrollArea
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # --- 核心修改: 使用 objectName 代替成员变量 ---
        scroll_area.setObjectName("projection_analysis_tab_widget")
        
        face_widget = QWidget()
        face_layout = QVBoxLayout()
        
        # 这部分 face_info 只是数据，不需要修改
        cube_size = self.cube_size_spin.value()
        face_info = [
            {'name': 'x=0', 'title': 'X=0面 (YZ平面)', 'axes': ['Y', 'Z'], 'indices': [1, 2]},
            {'name': f'x={cube_size}', 'title': f'X={cube_size}面 (YZ平面)', 'axes': ['Y', 'Z'], 'indices': [1, 2]},
            {'name': 'y=0', 'title': 'Y=0面 (XZ平面)', 'axes': ['X', 'Z'], 'indices': [0, 2]},
            {'name': f'y={cube_size}', 'title': f'Y={cube_size}面 (XZ平面)', 'axes': ['X', 'Z'], 'indices': [0, 2]},
            {'name': 'z=0', 'title': 'Z=0面 (XY平面)', 'axes': ['X', 'Y'], 'indices': [0, 1]},
            {'name': f'z={cube_size}', 'title': f'Z={cube_size}面 (XY平面)', 'axes': ['X', 'Y'], 'indices': [0, 1]}
        ]
        
        # --- 修改开始: 将所有 QLabel 的文本移除，并创建实例变量 ---
        
        # 第一行：六个面平均数据
        self.proj_avg_all_label = QLabel() # 修改
        self.proj_avg_all_label.setAlignment(Qt.AlignCenter)
        self.proj_avg_all_label.setStyleSheet("font-weight: bold; margin: 5px; font-size: 16px; color: #E74C3C; background-color: #FDF2F2; padding: 10px; border: 2px solid #E74C3C; border-radius: 5px;")
        face_layout.addWidget(self.proj_avg_all_label)
        
        average_row_widget = QWidget()
        average_row_layout = QHBoxLayout()
        average_row_widget.setLayout(average_row_layout)
        
        self.average_fractal_canvas = MplCanvas(self, width=6, height=6, dpi=100, is_3d=False)
        self.average_fractal_canvas.setFixedSize(600, 600)
        average_row_layout.addWidget(self.average_fractal_canvas)
        
        self.average_data_canvas = MplCanvas(self, width=3, height=6, dpi=100, is_3d=False)
        self.average_data_canvas.setFixedSize(600, 600)
        average_row_layout.addWidget(self.average_data_canvas)
        

        face_layout.addWidget(average_row_widget)
        
        separator1 = QFrame() # 给分隔符也起个名字
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        separator1.setStyleSheet("QFrame { background-color: #E74C3C; min-height: 3px; }")
        face_layout.addWidget(separator1)
        
        # 第二行：YOZ方向平均数据
        self.proj_avg_yoz_label = QLabel() # 修改
        self.proj_avg_yoz_label.setAlignment(Qt.AlignCenter)
        self.proj_avg_yoz_label.setStyleSheet("font-weight: bold; margin: 5px; font-size: 16px; color: #27AE60; background-color: #E8F8F5; padding: 10px; border: 2px solid #27AE60; border-radius: 5px;")
        face_layout.addWidget(self.proj_avg_yoz_label)
        
        yoz_row_widget = QWidget()
        yoz_row_layout = QHBoxLayout()
        yoz_row_widget.setLayout(yoz_row_layout)
        
        self.group_fractal_canvases['YOZ'] = MplCanvas(self, width=6, height=6, dpi=100, is_3d=False)
        self.group_fractal_canvases['YOZ'].setFixedSize(600, 600)
        yoz_row_layout.addWidget(self.group_fractal_canvases['YOZ'])
        
        self.group_data_canvases['YOZ'] = MplCanvas(self, width=3, height=6, dpi=100, is_3d=False)
        self.group_data_canvases['YOZ'].setFixedSize(600, 600)
        yoz_row_layout.addWidget(self.group_data_canvases['YOZ'])
        

        face_layout.addWidget(yoz_row_widget)
        
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet("QFrame { background-color: #27AE60; min-height: 3px; }")
        face_layout.addWidget(separator2)
        
        # 第三行：XOZ方向平均数据
        self.proj_avg_xoz_label = QLabel() # 修改
        self.proj_avg_xoz_label.setAlignment(Qt.AlignCenter)
        self.proj_avg_xoz_label.setStyleSheet("font-weight: bold; margin: 5px; font-size: 16px; color: #8E44AD; background-color: #F4ECF7; padding: 10px; border: 2px solid #8E44AD; border-radius: 5px;")
        face_layout.addWidget(self.proj_avg_xoz_label)
        
        xoz_row_widget = QWidget()
        xoz_row_layout = QHBoxLayout()
        xoz_row_widget.setLayout(xoz_row_layout)
        
        self.group_fractal_canvases['XOZ'] = MplCanvas(self, width=6, height=6, dpi=100, is_3d=False)
        self.group_fractal_canvases['XOZ'].setFixedSize(600, 600)
        xoz_row_layout.addWidget(self.group_fractal_canvases['XOZ'])
        
        self.group_data_canvases['XOZ'] = MplCanvas(self, width=3, height=6, dpi=100, is_3d=False)
        self.group_data_canvases['XOZ'].setFixedSize(600, 600)
        xoz_row_layout.addWidget(self.group_data_canvases['XOZ'])
        

        face_layout.addWidget(xoz_row_widget)
        
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        separator3.setStyleSheet("QFrame { background-color: #8E44AD; min-height: 3px; }")
        face_layout.addWidget(separator3)
        
        # 第四行：XOY方向平均数据
        self.proj_avg_xoy_label = QLabel() # 修改
        self.proj_avg_xoy_label.setAlignment(Qt.AlignCenter)
        self.proj_avg_xoy_label.setStyleSheet("font-weight: bold; margin: 5px; font-size: 16px; color: #F39C12; background-color: #FEF9E7; padding: 10px; border: 2px solid #F39C12; border-radius: 5px;")
        face_layout.addWidget(self.proj_avg_xoy_label)
        
        xoy_row_widget = QWidget()
        xoy_row_layout = QHBoxLayout()
        xoy_row_widget.setLayout(xoy_row_layout)
        
        self.group_fractal_canvases['XOY'] = MplCanvas(self, width=6, height=6, dpi=100, is_3d=False)
        self.group_fractal_canvases['XOY'].setFixedSize(600, 600)
        xoy_row_layout.addWidget(self.group_fractal_canvases['XOY'])
        
        self.group_data_canvases['XOY'] = MplCanvas(self, width=3, height=6, dpi=100, is_3d=False)
        self.group_data_canvases['XOY'].setFixedSize(600, 600)
        xoy_row_layout.addWidget(self.group_data_canvases['XOY'])
        

        face_layout.addWidget(xoy_row_widget)
        
        # --- 结束修改 ---
        
        face_widget.setLayout(face_layout)
        scroll_area.setWidget(face_widget)
        
        # 标签页的标题将由 retranslate_ui 动态设置
        self.tab_widget.addTab(scroll_area, "...") 
    
# 文件: FD-Advanced_3D.py


    def _create_statistics_analysis_tab(self):
        """创建统计分析标签页 (v3 - 修复3D/2D画布混淆错误)"""
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False) # 保持False，确保滚动条正常工作
        
        scroll_area.setObjectName("statistics_analysis_tab_widget")
        
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        stats_layout.setAlignment(Qt.AlignHCenter)

        FIXED_PLOT_WIDTH = 1000
        FIXED_PLOT_HEIGHT = 600
        
        # 倾角分布图
        self.inclination_label = QLabel() 
        self.inclination_label.setAlignment(Qt.AlignCenter)
        self.inclination_label.setStyleSheet("font-weight: bold; margin: 10px 0; font-size: 14px; color: #2E86AB;")
        stats_layout.addWidget(self.inclination_label)
        
        # --- 核心修正: 明确指定为2D画布 ---
        self.inclination_canvas = MplCanvas(self, width=10, height=6, dpi=100, is_3d=False)
        self.inclination_canvas.setFixedSize(FIXED_PLOT_WIDTH, FIXED_PLOT_HEIGHT)
        stats_layout.addWidget(self.inclination_canvas)
        
        # 方位角分布图
        self.azimuth_label = QLabel() 
        self.azimuth_label.setAlignment(Qt.AlignCenter)
        self.azimuth_label.setStyleSheet("font-weight: bold; margin: 10px 0; font-size: 14px; color: #2E86AB;")
        stats_layout.addWidget(self.azimuth_label)
        
        # --- 核心修正: 明确指定为2D画布 ---
        self.azimuth_canvas = MplCanvas(self, width=10, height=6, dpi=100, is_3d=False)
        self.azimuth_canvas.setFixedSize(FIXED_PLOT_WIDTH, FIXED_PLOT_HEIGHT)
        stats_layout.addWidget(self.azimuth_canvas)
        
        # 三维分形维数拟合曲线图 (注意：此图本身是2D的log-log图)
        self.fractal_3d_label = QLabel() 
        self.fractal_3d_label.setAlignment(Qt.AlignCenter)
        self.fractal_3d_label.setStyleSheet("font-weight: bold; margin: 10px 0; font-size: 14px; color: #2E86AB;")
        stats_layout.addWidget(self.fractal_3d_label)
        
        # --- 核心修正: 明确指定为2D画布 ---
        self.fractal_3d_canvas = MplCanvas(self, width=10, height=6, dpi=100, is_3d=False)
        self.fractal_3d_canvas.setFixedSize(FIXED_PLOT_WIDTH, FIXED_PLOT_HEIGHT)
        stats_layout.addWidget(self.fractal_3d_canvas)

        # “一键导出”按钮
        self.export_stats_btn = QPushButton()
        self.export_stats_btn.clicked.connect(self.export_statistics_plots)
        self.export_stats_btn.setEnabled(False) 
        self.export_stats_btn.setFixedWidth(400) 
        self.export_stats_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60; color: white; border: none;
                padding: 10px; font-size: 16px; font-weight: bold;
                border-radius: 5px; margin: 20px 0 10px 0;
            }
            QPushButton:hover { background-color: #229954; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        stats_layout.addWidget(self.export_stats_btn, alignment=Qt.AlignCenter)
        
        stats_widget.setLayout(stats_layout)
        scroll_area.setWidget(stats_widget)
        
        self.tab_widget.addTab(scroll_area, "...")

# 文件: FD-Advanced_3D.py (在 FractalFractureGeneratorGUI 类中)

    def export_statistics_plots(self):
        """
        将“统计分析”标签页中的三张图表分别保存到用户选择的目录中。
        """
        if not (hasattr(self, 'inclination_canvas') and self.inclination_canvas and 
                hasattr(self, 'azimuth_canvas') and self.azimuth_canvas and 
                hasattr(self, 'fractal_3d_canvas') and self.fractal_3d_canvas):
            QMessageBox.warning(self, self.tr("警告"), self.tr("图表尚未生成，无法导出。"))
            return

        # 1. 让用户选择一个目录
        directory = QFileDialog.getExistingDirectory(self, self.tr("选择导出目录"))
        
        if not directory:
            return  # 用户取消了选择

        try:
            # 2. 创建带时间戳的文件名前缀，避免覆盖
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 3. 定义三张图的完整文件路径
            path_inclination = os.path.join(directory, f"stats_inclination_{timestamp}.png")
            path_azimuth = os.path.join(directory, f"stats_azimuth_{timestamp}.png")
            path_fractal = os.path.join(directory, f"stats_3d_fractal_fit_{timestamp}.png")
            
            plots_to_save = {
                path_inclination: self.inclination_canvas.fig,
                path_azimuth: self.azimuth_canvas.fig,
                path_fractal: self.fractal_3d_canvas.fig
            }

            # 4. 循环保存每张图，DPI设置为500
            for path, figure in plots_to_save.items():
                # 在保存前再次调用 tight_layout 确保布局紧凑
                figure.tight_layout()
                figure.savefig(path, dpi=500, bbox_inches='tight')
            
            # --- 新增：导出数据到TXT ---
            data_file_path = os.path.join(directory, f"fracture_analysis_data_{timestamp}.txt")
            with open(data_file_path, 'w', encoding='utf-8') as f:
                f.write(f"===== fracture analysis data export =====\n")
                f.write(f"Export Time: {timestamp}\n")
                f.write(f"Cube Size: {self.cube_size_spin.value()}\n")
                f.write(f"Fracture Count: {len(self.latest_fractures)}\n\n")

                # 1. Fractal Dimension Data
                if self.latest_levels:
                    f.write(f"----- 3D Fractal Dimension Analysis -----\n")
                    f.write(f"Comparison: Expected D={self.fractal_dim_spin.value()}, Calculated D={self.latest_calculated_dim:.4f}\n")
                    f.write(f"{'Box Size':<15}\t{'Count'}\n")
                    for level in self.latest_levels:
                         f.write(f"{level['box_size']:<15.6f}\t{level['valid_count']}\n")
                    f.write("\n")

                # 2. Orientation Data
                f.write(f"----- Orientation Data (Degrees) -----\n")
                f.write(f"{'ID':<10}\t{'Inclination':<15}\t{'Azimuth'}\n")
                for i, fracture in enumerate(self.latest_fractures):
                    inc_deg = np.rad2deg(fracture.inclination)
                    az_deg = np.rad2deg(fracture.azimuth)
                    f.write(f"{i:<10}\t{inc_deg:<15.4f}\t{az_deg:.4f}\n")
            
            # 5. 显示成功信息
            success_message = self.tr("3 statistics charts and 1 data file have been successfully exported to directory:\n{0}").format(directory)
            QMessageBox.information(self, self.tr("导出成功"), success_message)

        except Exception as e:
            error_message = self.tr("导出图表时发生错误:\n{0}").format(str(e))
            QMessageBox.critical(self, self.tr("导出失败"), error_message)
    
    def calculate_and_plot_face_views(self, show_progress=True):
        """计算6个面的分形维数数据并绘制平均数据"""
        if not self.latest_fractures:
            print("没有裂缝数据，跳过六个面投影轨迹分析")
            return
            
        cube_size = self.cube_size_spin.value()
        num_slices = self.main_num_slices_spin.value() # <--- 从UI控件获取值

        
        if show_progress and hasattr(self, 'stats_text'):
            # --- 核心修改：翻译进度消息 ---
            self.stats_text.setText(self.tr("正在计算六个面分形维数数据..."))
            QApplication.processEvents()
        
        if hasattr(self, 'average_fractal_canvas'): self.average_fractal_canvas.axes.clear()
        if hasattr(self, 'average_data_canvas'): self.average_data_canvas.axes.clear()
        
        print(f"开始计算{3 * num_slices}个面分形维数数据，共有{len(self.latest_fractures)}个裂缝")
        
        face_configs = []
        num_slices = self.main_num_slices_spin.value() # <--- 使用UI控件的值
        for i in range(num_slices):
            x_value = i * (1.0 / (num_slices - 1)) * cube_size if num_slices > 1 else 0.5 * cube_size
            face_configs.append({'name': f'x={x_value:.3f}', 'coord': 0, 'value': x_value, 'axes': ['Y', 'Z'], 'indices': [1, 2]})
        for i in range(num_slices):
            y_value = i * (1.0 / (num_slices - 1)) * cube_size if num_slices > 1 else 0.5 * cube_size
            face_configs.append({'name': f'y={y_value:.3f}', 'coord': 1, 'value': y_value, 'axes': ['X', 'Z'], 'indices': [0, 2]})
        for i in range(num_slices):
            z_value = i * (1.0 / (num_slices - 1)) * cube_size if num_slices > 1 else 0.5 * cube_size
            face_configs.append({'name': f'z={z_value:.3f}', 'coord': 2, 'value': z_value, 'axes': ['X', 'Y'], 'indices': [0, 1]})
        
        all_faces_data = []
        
        for i, face_config in enumerate(face_configs):
            if show_progress and hasattr(self, 'stats_text'):
                # --- 核心修改：翻译进度消息 ---
                progress_msg = self.tr("正在处理面 {0} ({1}/{2})...").format(face_config['name'], i + 1, len(face_configs))
                self.stats_text.setText(progress_msg)
                QApplication.processEvents()
            
            face_line_segments = []
            for fracture in self.latest_fractures:
                if hasattr(fracture, 'boundary_lines') and len(fracture.boundary_lines) > 0:
                    for line_info in fracture.boundary_lines:
                        if line_info['face_name'] == face_config['name']:
                            points = line_info['points']
                            if len(points) >= 2:
                                x_coords = points[:, face_config['indices'][0]]
                                y_coords = points[:, face_config['indices'][1]]
                                line_coords = [(x_coords[j], y_coords[j]) for j in range(len(x_coords))]
                                face_line_segments.append(line_coords)
            
            fractal_dim, fit_data, levels = self._calculate_fractal_dimension_for_face(face_line_segments, cube_size, face_config['name'])
            
            if fractal_dim is not None and fit_data is not None:
                all_faces_data.append({'face_name': face_config['name'], 'fractal_dim': fractal_dim, 'fit_data': fit_data, 'levels': levels})
        
        if len(all_faces_data) > 0: self._plot_average_fractal_dimension(all_faces_data, cube_size, num_slices)
        if len(all_faces_data) >= 6: self._plot_group_average_fractal_dimensions(all_faces_data, cube_size, num_slices)

        self.latest_face_analysis_data = all_faces_data

    
        # <<< 在 calculate_and_plot_face_views 函数结束后，粘贴这个新函数 >>>
    
    def plot_face_views(self):
        """
        快速重绘函数：只使用已保存的数据进行绘图，不重新计算。
        """
        if not self.latest_face_analysis_data:
            # 如果还没有计算过数据，就什么都不做
            return
        
        cube_size = self.cube_size_spin.value()
        num_slices = self.main_num_slices_spin.value() # <--- 从UI获取值

        
        # 直接调用绘图函数，并传入已保存的数据
        self._plot_average_fractal_dimension(self.latest_face_analysis_data, cube_size, num_slices)
        self._plot_group_average_fractal_dimensions(self.latest_face_analysis_data, cube_size, num_slices)

    
    def _calculate_fractal_dimension_for_face(self, line_segments, cube_size, face_name):
            """为单个面计算分形维数（不绘制图形）"""
            if len(line_segments) == 0:
                print(f"{face_name}面无裂缝数据，无法计算分形维数")
                return None, None, None
            
            # 获取计算迭代次数
            calc_iterations = self.calc_iterations_spin.value()
            
            # 使用新的、正确的构造函数创建实例
            temp_calculator = FractalDimension2DCalculator(iterations=calc_iterations) # <--- 修改后的代码
            
            # 传递回调函数（如果需要）
            if hasattr(self.fractal_2d_calculator, 'progress_callback'):
                temp_calculator.progress_callback = self.fractal_2d_calculator.progress_callback

            # calculate_fractal_dimension 方法现在只需要轨迹、宽度和高度
            # 对于立方体的面，宽度和高度都是 cube_size
            fractal_dim, fit_data, levels = temp_calculator.calculate_fractal_dimension(
                line_segments, cube_size, cube_size
            )
            
            if fractal_dim is None or fit_data is None:
                print(f"{face_name}面数据不足，无法计算分形维数")
                return None, None, None
            
            # r_squared 现在直接在 fit_data 字典里
            r_squared = fit_data['r_squared']
            
            return fractal_dim, fit_data, levels
    
    def _plot_fractal_dimension_for_face(self, line_segments, cube_size, canvas, data_canvas, face_name):
        """为单个面计算并绘制分形维数"""
        # 清除画布
        canvas.axes.clear()
        data_canvas.axes.clear()
        
        if len(line_segments) == 0:
            # 没有裂缝数据
            canvas.axes.text(0.5, 0.5, '无裂缝数据\n无法计算分形维数', 
                           transform=canvas.axes.transAxes, ha='center', va='center', 
                           fontsize=12, fontfamily='Microsoft YaHei',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            canvas.axes.set_title(f'{face_name}面分形维数', fontfamily='Microsoft YaHei')
            canvas.axes.set_xlabel(r'$\log_{10}$(盒子尺寸)', fontfamily='Microsoft YaHei')
            canvas.axes.set_ylabel(r'$\log_{10}$(裂缝数量)', fontfamily='Microsoft YaHei')
            
            # 数据显示图也显示无数据
            data_canvas.axes.text(0.5, 0.5, '无裂缝数据', 
                                 transform=data_canvas.axes.transAxes, ha='center', va='center', 
                                 fontsize=12, fontfamily='Microsoft YaHei',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            data_canvas.axes.set_title(f'{face_name}面数据', fontfamily='Microsoft YaHei')
            data_canvas.axes.axis('off')
            return None, None, None
        
        # 获取计算迭代次数并计算分形维数
        calc_iterations = self.calc_iterations_spin.value()
        # 创建新的2D分形维数计算器实例，使用指定的迭代次数
        gen_iterations = self.iterations_spin.value()
        temp_calculator = FractalDimension2DCalculator(iterations=calc_iterations, generation_iterations=gen_iterations)
        temp_calculator.progress_callback = self.fractal_2d_calculator.progress_callback
        fractal_dim, fit_data, levels = temp_calculator.calculate_fractal_dimension(
            line_segments, cube_size, face_name
        )
        
        if fractal_dim is None or fit_data is None:
            # 计算失败
            canvas.axes.text(0.5, 0.5, '数据不足\n无法计算分形维数', 
                           transform=canvas.axes.transAxes, ha='center', va='center', 
                           fontsize=12, fontfamily='Microsoft YaHei',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
            canvas.axes.set_title(f'{face_name}面分形维数', fontfamily='Microsoft YaHei')
            canvas.axes.set_xlabel(r'$\log_{10}$(盒子尺寸)', fontfamily='Microsoft YaHei')
            canvas.axes.set_ylabel(r'$\log_{10}$(裂缝数量)', fontfamily='Microsoft YaHei')
            
            # 数据显示图也显示计算失败
            data_canvas.axes.text(0.5, 0.5, '数据不足\n无法计算分形维数', 
                                 transform=data_canvas.axes.transAxes, ha='center', va='center', 
                                 fontsize=12, fontfamily='Microsoft YaHei',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
            data_canvas.axes.set_title(f'{face_name}面数据', fontfamily='Microsoft YaHei')
            data_canvas.axes.axis('off')
            return None, None, None
        
        # 获取拟合数据
        log_sizes = fit_data['log_sizes']
        log_counts = fit_data['log_counts']
        coeffs = fit_data['coeffs']
        r_squared = fit_data['r_squared']
        
        # 计算坐标轴范围
        x_min, x_max = np.min(log_sizes), np.max(log_sizes)
        y_min, y_max = np.min(log_counts), np.max(log_counts)
        
        # 增加边距
        x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 0.5
        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.5
        
        # 设置坐标轴范围
        canvas.axes.set_xlim(x_min - x_margin, x_max + x_margin)
        canvas.axes.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # 绘制散点图
        canvas.axes.scatter(log_sizes, log_counts, color='red', s=60, alpha=0.8, zorder=3)
        
        # 绘制拟合直线
        x_fit = np.linspace(x_min - x_margin, x_max + x_margin, 100)
        y_fit = np.polyval(coeffs, x_fit)
        canvas.axes.plot(x_fit, y_fit, 'b--', linewidth=2, zorder=2)
        
        # 设置坐标轴标签（使用数学表达式）
        canvas.axes.set_xlabel(r'$\log_{10}$(盒子尺寸)', fontsize=10, fontfamily='Microsoft YaHei')
        canvas.axes.set_ylabel(r'$\log_{10}$(裂缝数量)', fontsize=10, fontfamily='Microsoft YaHei')
        canvas.axes.grid(True, alpha=0.3)
        
        # 计算N0值
        calculated_n0 = 10 ** coeffs[1]
        
        # 设置标题，包含分形维数信息，将N0写在同一行
        title = f'{face_name}面分形维数\nD = {fractal_dim:.3f}, $N_0$ = {calculated_n0:.1f}'
        if r_squared > 0:
            title += f', R² = {r_squared:.3f}'
        canvas.axes.set_title(title, fontsize=11, pad=10, fontfamily='Microsoft YaHei')
        
        # 在数据显示画布中显示L和N数据
        data_text = "L / N\n"
        for level in levels:
            data_text += f"{level['box_size']:.3f} / {level['valid_count']}\n"
        
        # 将数据信息显示在数据画布中央
        data_canvas.axes.text(0.5, 0.5, data_text, transform=data_canvas.axes.transAxes,
                             fontsize=20, ha='center', va='center', fontfamily='Microsoft YaHei',
                             )
        data_canvas.axes.set_title(f'{face_name}面数据', fontfamily='Microsoft YaHei')
        data_canvas.axes.axis('off')
        
        return fractal_dim, fit_data, levels
    
# 请将这个完整的函数块粘贴到原来的位置

    # 这是针对您最新代码的修复版本
    def _plot_average_fractal_dimension(self, all_faces_data, cube_size, num_slices):
        """计算并绘制六个面的平均分形维数数据"""
        if not hasattr(self, 'average_fractal_canvas') or not hasattr(self, 'average_data_canvas'):
            return
            
        self.average_fractal_canvas.axes.clear()
        self.average_data_canvas.axes.clear()
        num_slices_total = 3 * num_slices 

        
        if len(all_faces_data) == 0:
            self.average_fractal_canvas.axes.text(0.5, 0.5, self.tr('无有效数据\n无法计算平均分形维数'), 
                                            transform=self.average_fractal_canvas.axes.transAxes, ha='center', va='center', 
                                            fontsize=12, fontfamily='Microsoft YaHei',
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            self.average_fractal_canvas.axes.set_title(self.tr('{0}个面平均分形维数').format(num_slices_total), fontfamily='Microsoft YaHei')
            
            x_label_text = self.tr("盒子尺寸")
            y_label_text = self.tr("裂缝数量")
            self.average_fractal_canvas.axes.set_xlabel(rf'$\log_{{10}}$({x_label_text})', fontfamily='Microsoft YaHei')
            self.average_fractal_canvas.axes.set_ylabel(rf'$\log_{{10}}$({y_label_text})', fontfamily='Microsoft YaHei')
            
            self.average_data_canvas.axes.text(0.5, 0.5, self.tr('无有效数据'), 
                                            transform=self.average_data_canvas.axes.transAxes, ha='center', va='center', 
                                            fontsize=12, fontfamily='Microsoft YaHei',
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            self.average_data_canvas.axes.set_title(self.tr('全部面平均数据'), fontfamily='Microsoft YaHei')
            self.average_data_canvas.axes.axis('off')
            self.average_fractal_canvas.draw()
            self.average_data_canvas.draw()
            return
        
        fractal_dims = [data['fractal_dim'] for data in all_faces_data]
        n0_values = [10 ** data['fit_data']['coeffs'][1] for data in all_faces_data]
        average_n0 = np.mean(n0_values)
        r_squared_values = [data['fit_data']['r_squared'] for data in all_faces_data]
        average_r_squared = np.mean(r_squared_values)
        
        all_levels_data = {}
        for face_data in all_faces_data:
            for level in face_data['levels']:
                box_size = level['box_size']
                if box_size not in all_levels_data:
                    all_levels_data[box_size] = []
                all_levels_data[box_size].append(level['valid_count'])
        
        average_levels = []
        for box_size in sorted(all_levels_data.keys(), reverse=True):
            avg_count = np.mean(all_levels_data[box_size])
            average_levels.append({
                'box_size': box_size,
                'valid_count': avg_count
            })
        
        filtered_average_levels = [level for level in average_levels if level['valid_count'] > 0]
        if len(filtered_average_levels) < 2:
            self.average_fractal_canvas.axes.text(0.5, 0.5, self.tr('有效数据点不足\n无法计算平均分形维数'), 
                                            transform=self.average_fractal_canvas.axes.transAxes, ha='center', va='center', 
                                            fontsize=12, fontfamily='Microsoft YaHei',
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
            self.average_fractal_canvas.axes.set_title(self.tr('{0}个面平均分形维数').format(num_slices_total), fontfamily='Microsoft YaHei')
            
            x_label_text = self.tr("盒子尺寸")
            y_label_text = self.tr("裂缝数量")
            self.average_fractal_canvas.axes.set_xlabel(rf'$\log_{{10}}$({x_label_text})', fontfamily='Microsoft YaHei')
            self.average_fractal_canvas.axes.set_ylabel(rf'$\log_{{10}}$({y_label_text})', fontfamily='Microsoft YaHei')
            
            self.average_data_canvas.axes.text(0.5, 0.5, self.tr('有效数据点不足'), 
                                            transform=self.average_data_canvas.axes.transAxes, ha='center', va='center', 
                                            fontsize=12, fontfamily='Microsoft YaHei',
                                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
            self.average_data_canvas.axes.set_title(self.tr('全部面平均数据'), fontfamily='Microsoft YaHei')
            self.average_data_canvas.axes.axis('off')
            self.average_fractal_canvas.draw()
            self.average_data_canvas.draw()
            print(self.tr("{0}个面平均分形维数计算失败: 有效数据点不足").format(num_slices_total))
            return
        
        log_sizes = [np.log10(level['box_size']) for level in filtered_average_levels]
        log_counts = [np.log10(level['valid_count']) for level in filtered_average_levels]
        
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fitted_dim = -coeffs[0]
        
        y_fit_vals = np.polyval(coeffs, log_sizes)
        ss_res = np.sum((log_counts - y_fit_vals) ** 2)
        ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
        r_squared_fit = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        x_min, x_max = np.min(log_sizes), np.max(log_sizes)
        y_min, y_max = np.min(log_counts), np.max(log_counts)
        x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 0.5
        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.5
        self.average_fractal_canvas.axes.set_xlim(x_min - x_margin, x_max + x_margin)
        self.average_fractal_canvas.axes.set_ylim(y_min - y_margin, y_max + y_margin)
        
        self.average_fractal_canvas.axes.scatter(log_sizes, log_counts, color='red', s=60, alpha=0.8, zorder=3)
        
        x_fit = np.linspace(x_min - x_margin, x_max + x_margin, 100)
        y_fit = np.polyval(coeffs, x_fit)
        self.average_fractal_canvas.axes.plot(x_fit, y_fit, 'b--', linewidth=2, zorder=2)
        
        x_label_text = self.tr("盒子尺寸")
        y_label_text = self.tr("裂缝数量")
        self.average_fractal_canvas.axes.set_xlabel(rf'$\log_{{10}}$({x_label_text})', fontsize=10, fontfamily='Microsoft YaHei')
        self.average_fractal_canvas.axes.set_ylabel(rf'$\log_{{10}}$({y_label_text})', fontsize=10, fontfamily='Microsoft YaHei')
        self.average_fractal_canvas.axes.grid(True, alpha=0.3)
        
        title = self.tr('{0}个面平均分形维数').format(num_slices_total) + f'\nD = {fitted_dim:.3f}, $N_0$ = {average_n0:.1f}'
        if r_squared_fit > 0:
            title += f', R² = {r_squared_fit:.3f}'
        self.average_fractal_canvas.axes.set_title(title, fontsize=11, pad=10, fontfamily='Microsoft YaHei')
        
        data_title_text = self.tr('全部面平均数据')
        data_header_text = self.tr("L / N (平均)")
        data_dim_text = self.tr("平均counts的分形维数")
        
        data_text = f"{data_header_text}\n"
        for level in filtered_average_levels:
            data_text += f"{level['box_size']:.3f} / {level['valid_count']:.3f}\n"
        data_text += f"{data_dim_text}: {fitted_dim:.3f}\n"
        data_text += f"R²: {average_r_squared:.3f}\n"
        
        self.average_data_canvas.axes.text(0.5, 0.5, data_text, transform=self.average_data_canvas.axes.transAxes,
                                        fontsize=12, ha='center', va='center', fontfamily='Microsoft YaHei')
        self.average_data_canvas.axes.set_title(data_title_text, fontfamily='Microsoft YaHei')
        self.average_data_canvas.axes.axis('off')
        
        self.average_fractal_canvas.draw()
        self.average_data_canvas.draw()
        
        print(self.tr("{0}个面平均分形维数计算完成: 平均counts的D = {1:.4f}, R² = {2:.4f}").format(num_slices_total, fitted_dim, r_squared_fit))
            
# 请将这个完整的函数块粘贴到原来的位置

    # 请将这个完整的函数块粘贴到原来的位置

    def _plot_group_average_fractal_dimensions(self, all_faces_data, cube_size, num_slices):
        """计算并绘制YOZ、XOZ、XOY三组的平均分形维数数据"""
        num_slices = self.main_num_slices_spin.value() # <--- 使用UI控件的值
        groups = {
            'YOZ': [f'x={i * (1.0 / (num_slices - 1)) * cube_size:.3f}' for i in range(num_slices)],
            'XOZ': [f'y={i * (1.0 / (num_slices - 1)) * cube_size:.3f}' for i in range(num_slices)],
            'XOY': [f'z={i * (1.0 / (num_slices - 1)) * cube_size:.3f}' for i in range(num_slices)]
        }
        
        for group_name, face_names in groups.items():
            if not hasattr(self, 'group_fractal_canvases') or group_name not in self.group_fractal_canvases:
                continue
                
            fractal_canvas = self.group_fractal_canvases[group_name]
            data_canvas = self.group_data_canvases[group_name]
            
            fractal_canvas.axes.clear()
            data_canvas.axes.clear()
            
            group_faces_data = [data for data in all_faces_data if data['face_name'] in face_names]
            
            if len(group_faces_data) < num_slices:
                fractal_canvas.axes.text(0.5, 0.5, self.tr('{0}组数据不足\n无法计算平均分形维数').format(group_name), 
                                    transform=fractal_canvas.axes.transAxes, ha='center', va='center', 
                                    fontsize=12, fontfamily='Microsoft YaHei',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                fractal_canvas.axes.set_title(self.tr('{0}组平均分形维数').format(group_name), fontfamily='Microsoft YaHei')
                x_label_text = self.tr("盒子尺寸")
                y_label_text = self.tr("裂缝数量")
                fractal_canvas.axes.set_xlabel(rf'$\log_{{10}}$({x_label_text})', fontfamily='Microsoft YaHei')
                fractal_canvas.axes.set_ylabel(rf'$\log_{{10}}$({y_label_text})', fontfamily='Microsoft YaHei')
                
                data_canvas.axes.text(0.5, 0.5, self.tr('{0}组数据不足').format(group_name), 
                                    transform=data_canvas.axes.transAxes, ha='center', va='center', 
                                    fontsize=12, fontfamily='Microsoft YaHei',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                data_canvas.axes.set_title(self.tr('{0}组平均数据').format(group_name), fontfamily='Microsoft YaHei')
                data_canvas.axes.axis('off')
                
                fractal_canvas.draw()
                data_canvas.draw()
                continue
            
            group_fractal_dims = [data['fractal_dim'] for data in group_faces_data]
            group_n0_values = [10 ** data['fit_data']['coeffs'][1] for data in group_faces_data]
            group_average_n0 = np.mean(group_n0_values)
            group_r_squared_values = [data['fit_data']['r_squared'] for data in group_faces_data]
            
            group_levels_data = {}
            for face_data in group_faces_data:
                for level in face_data['levels']:
                    box_size = level['box_size']
                    if box_size not in group_levels_data:
                        group_levels_data[box_size] = []
                    group_levels_data[box_size].append(level['valid_count'])
            
            group_average_levels = []
            for box_size in sorted(group_levels_data.keys(), reverse=True):
                avg_count = np.mean(group_levels_data[box_size])
                group_average_levels.append({
                    'box_size': box_size,
                    'valid_count': avg_count
                })
            
            filtered_group_levels = [level for level in group_average_levels if level['valid_count'] > 0]
            if len(filtered_group_levels) < 2:
                fractal_canvas.axes.text(0.5, 0.5, self.tr('{0}组有效数据点不足\n无法计算平均分形维数').format(group_name), 
                                    transform=fractal_canvas.axes.transAxes, ha='center', va='center', 
                                    fontsize=12, fontfamily='Microsoft YaHei',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
                fractal_canvas.axes.set_title(self.tr('{0}组平均分形维数').format(group_name), fontfamily='Microsoft YaHei')
                x_label_text = self.tr("盒子尺寸")
                y_label_text = self.tr("裂缝数量")
                fractal_canvas.axes.set_xlabel(rf'$\log_{{10}}$({x_label_text})', fontfamily='Microsoft YaHei')
                fractal_canvas.axes.set_ylabel(rf'$\log_{{10}}$({y_label_text})', fontfamily='Microsoft YaHei')
                
                data_canvas.axes.text(0.5, 0.5, self.tr('{0}组有效数据点不足').format(group_name), 
                                    transform=data_canvas.axes.transAxes, ha='center', va='center', 
                                    fontsize=12, fontfamily='Microsoft YaHei',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
                data_canvas.axes.set_title(self.tr('{0}组平均数据').format(group_name), fontfamily='Microsoft YaHei')
                data_canvas.axes.axis('off')
                
                fractal_canvas.draw()
                data_canvas.draw()
                continue
            
            log_sizes = [np.log10(level['box_size']) for level in filtered_group_levels]
            log_counts = [np.log10(level['valid_count']) for level in filtered_group_levels]
            
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            fitted_dim = -coeffs[0]
            
            y_fit_vals = np.polyval(coeffs, log_sizes)
            ss_res = np.sum((log_counts - y_fit_vals) ** 2)
            ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
            r_squared_fit = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            x_min, x_max = np.min(log_sizes), np.max(log_sizes)
            y_min, y_max = np.min(log_counts), np.max(log_counts)
            x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 0.5
            y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.5
            fractal_canvas.axes.set_xlim(x_min - x_margin, x_max + x_margin)
            fractal_canvas.axes.set_ylim(y_min - y_margin, y_max + y_margin)
            
            fractal_canvas.axes.scatter(log_sizes, log_counts, color='red', s=60, alpha=0.8, zorder=3)
            
            x_fit = np.linspace(x_min - x_margin, x_max + x_margin, 100)
            y_fit = np.polyval(coeffs, x_fit)
            fractal_canvas.axes.plot(x_fit, y_fit, 'b--', linewidth=2, zorder=2)
            
            x_label_text = self.tr("盒子尺寸")
            y_label_text = self.tr("裂缝数量")
            fractal_canvas.axes.set_xlabel(rf'$\log_{{10}}$({x_label_text})', fontsize=10, fontfamily='Microsoft YaHei')
            fractal_canvas.axes.set_ylabel(rf'$\log_{{10}}$({y_label_text})', fontsize=10, fontfamily='Microsoft YaHei')
            fractal_canvas.axes.grid(True, alpha=0.3)
            
            title = self.tr('{0}组平均分形维数').format(group_name) + f'\nD = {fitted_dim:.3f}, $N_0$ = {group_average_n0:.1f}'
            if r_squared_fit > 0:
                title += f', R² = {r_squared_fit:.3f}'
            fractal_canvas.axes.set_title(title, fontsize=11, pad=10, fontfamily='Microsoft YaHei')
            
            data_title_text = self.tr('{0}组平均数据').format(group_name)
            data_header_text = self.tr("L / N ({0}组平均)").format(group_name)
            data_dim_text = self.tr("平均counts的分形维数")
            stats_title_text = self.tr("{0}组统计信息:").format(group_name)
            stats_faces_text = self.tr("包含面数:")
            stats_points_text = self.tr("有效数据点:")

            data_text = f"{data_header_text}\n"
            for level in filtered_group_levels:
                data_text += f"{level['box_size']:.3f} / {level['valid_count']:.3f}\n"
            data_text += f"{data_dim_text}: {fitted_dim:.3f}\n"
            data_text += f"R²: {r_squared_fit:.3f}\n\n"
            data_text += f"{stats_title_text}\n"
            data_text += f"{stats_faces_text} {len(group_faces_data)}\n"
            data_text += f"{stats_points_text} {len(filtered_group_levels)}\n"
            
            data_canvas.axes.text(0.5, 0.5, data_text, transform=data_canvas.axes.transAxes,
                                fontsize=12, ha='center', va='center', fontfamily='Microsoft YaHei')
            data_canvas.axes.set_title(data_title_text, fontfamily='Microsoft YaHei')
            data_canvas.axes.axis('off')
            
            fractal_canvas.draw()
            data_canvas.draw()
            
            print(self.tr("{0}组平均分形维数计算完成: D = {1:.4f}, R² = {2:.4f}").format(group_name, fitted_dim, r_squared_fit))

# 文件: FD-Advanced_3D.py

# 文件: FD-Advanced_3D.py

    def plot_statistics_analysis(self, show_progress=True):
        """绘制统计分析图表（优化版本）"""
        if not self.latest_fractures or not self.inclination_canvas:
            return
        
        if show_progress and hasattr(self, 'stats_text'):
            self.stats_text.setText(self.tr("正在绘制统计分析图表..."))
            QApplication.processEvents()
        
        self.inclination_canvas.axes.clear()
        self.azimuth_canvas.axes.clear()
        self.fractal_3d_canvas.axes.clear()
        
        inclinations = [np.rad2deg(f.inclination) for f in self.latest_fractures]
        azimuths = [np.rad2deg(f.azimuth) for f in self.latest_fractures]
        
        if show_progress and hasattr(self, 'stats_text'):
            self.stats_text.setText(self.tr("正在绘制倾角分布图..."))
            QApplication.processEvents()
        self._plot_angle_distribution(self.inclination_canvas, inclinations, self.tr("倾角"), 0, 180)
        
        if show_progress and hasattr(self, 'stats_text'):
            self.stats_text.setText(self.tr("正在绘制方位角分布图..."))
            QApplication.processEvents()
        self._plot_angle_distribution(self.azimuth_canvas, azimuths, self.tr("方位角"), 0, 360)
        
        if self.latest_calculated_dim is not None and self.latest_fit_data is not None:
            self._plot_3d_fractal_comparison_with_data()
        else:
            self._plot_3d_fractal_comparison()
        
        self.inclination_canvas.draw()
        self.azimuth_canvas.draw()
        self.fractal_3d_canvas.draw()
        
        # --- 新增：图表绘制完成后，启用导出按钮 ---
        if hasattr(self, 'export_stats_btn'):
            self.export_stats_btn.setEnabled(True)

        QApplication.processEvents()
    
# 文件: FD-Advanced_3D.py

    def _plot_angle_distribution(self, canvas, angles, angle_type_text, min_angle, max_angle): # 修改: 参数名改为 angle_type_text
        """绘制角度分布柱状图"""
        if len(angles) == 0:
            # --- 修改: 使用 tr() ---
            canvas.axes.text(0.5, 0.5, self.tr('无{0}数据').format(angle_type_text), 
                           transform=canvas.axes.transAxes, ha='center', va='center', 
                           fontsize=12, fontfamily='Microsoft YaHei')
            return
        
        # 定义角度区间（每10度一个区间）
        bin_size = 10
        bins = np.arange(min_angle, max_angle + bin_size, bin_size)
        
        # 计算每个区间的裂缝数量
        counts, bin_edges = np.histogram(angles, bins=bins)
        
        # 创建区间标签
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            bin_labels.append(f'{int(bin_edges[i])}-{int(bin_edges[i+1])}°')
        
        # 绘制柱状图
        x_pos = np.arange(len(bin_labels))
        bars = canvas.axes.bar(x_pos, counts, alpha=0.7, color='steelblue', edgecolor='black')
        
        # 在柱子上标注数量
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count > 0:
                canvas.axes.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                str(int(count)), ha='center', va='bottom', fontsize=8)
        
        # 设置坐标轴
        canvas.axes.set_xlabel(self.tr('{0}范围').format(angle_type_text), fontfamily='Microsoft YaHei')
        canvas.axes.set_ylabel(self.tr('裂缝数量'), fontfamily='Microsoft YaHei')
        title = self.tr('{0}分布统计 (总计：{1}个裂缝)').format(angle_type_text, len(angles))
        canvas.axes.set_title(title, fontfamily='Microsoft YaHei')

        canvas.axes.set_xticks(x_pos)
        canvas.axes.set_xticklabels(bin_labels, rotation=45, ha='right')
        canvas.axes.grid(True, alpha=0.3, axis='y')
        
        # 设置Y轴范围
        if np.max(counts) > 0:
            canvas.axes.set_ylim(0, np.max(counts) * 1.1)

        # --- 新增：调用tight_layout()来自动调整布局，防止标签被裁切 ---
        canvas.fig.tight_layout()
    
    def _plot_3d_fractal_comparison(self):
        """绘制三维分形维数对比图（初始状态）"""
        # --- 修改 ---
        self.fractal_3d_canvas.axes.text(0.5, 0.5, self.tr('请先计算分形维数'), 
                                        transform=self.fractal_3d_canvas.axes.transAxes, 
                                        ha='center', va='center', fontsize=12, 
                                        fontfamily='Microsoft YaHei')
        self.fractal_3d_canvas.draw() # 添加 draw() 以确保更新
    
# 文件: FD-Advanced_3D.py

    def _plot_3d_fractal_comparison_with_data(self):
        """绘制三维分形维数对比图（包含数据）"""
        if not self.fractal_3d_canvas:
            return
        
        # <<< 核心修改：从 self 获取保存的数据 >>>
        calculated_dim = self.latest_calculated_dim
        fit_data = self.latest_fit_data
        levels = self.latest_levels
        # <<< 修改结束 >>>

        # 清空画布
        self.fractal_3d_canvas.axes.clear()
        
        # 获取理论参数
        expected_dim = self.fractal_dim_spin.value()
        expected_n0 = self.n0_spin.value()
        cube_size = self.cube_size_spin.value()
        
        # 生成理论曲线数据
        theory_box_sizes = [level['box_size'] for level in levels]
        theory_counts = [expected_n0 * (size ** (-expected_dim)) for size in theory_box_sizes]
        
        # 获取实际计算数据
        actual_box_sizes = [level['box_size'] for level in levels]
        actual_counts = [level['valid_count'] for level in levels]
        
        # 过滤掉计数为0的数据点用于绘制拟合线
        valid_indices = [i for i, count in enumerate(actual_counts) if count > 0]
        if len(valid_indices) >= 2:
            valid_sizes = [actual_box_sizes[i] for i in valid_indices]
            valid_counts = [actual_counts[i] for i in valid_indices]
            
            # 转换为对数坐标
            log_sizes = np.log10(valid_sizes)
            log_counts = np.log10(valid_counts)
            log_theory_sizes = np.log10(theory_box_sizes)
            log_theory_counts = np.log10(theory_counts)
            
            # 绘制散点图
            self.fractal_3d_canvas.axes.scatter(log_sizes, log_counts, color='red', s=70, marker='s',
                                              alpha=0.8, zorder=3, label=self.tr('实际计算值'))
            self.fractal_3d_canvas.axes.scatter(log_theory_sizes, log_theory_counts, color='blue', s=60, 
                                              alpha=0.8, zorder=3, label=self.tr('理论值'))
            
            # 绘制拟合直线
            coeffs = fit_data['coeffs']
            x_fit = np.linspace(min(log_sizes), max(log_sizes), 100)
            y_fit = np.polyval(coeffs, x_fit)
            self.fractal_3d_canvas.axes.plot(x_fit, y_fit, 'r--', linewidth=2, 
                                           label=self.tr('实际拟合线 (D={0:.3f})').format(calculated_dim))
            
            # 绘制理论直线
            theory_slope = -expected_dim
            theory_intercept = np.log10(expected_n0)
            y_theory = theory_slope * x_fit + theory_intercept
            self.fractal_3d_canvas.axes.plot(x_fit, y_theory, 'b-', linewidth=2, 
                                           label=self.tr('理论直线 (D={0:.3f})').format(expected_dim))
            
            # 设置坐标轴标签和标题
            # 1. 先翻译纯文本
            x_label_text = self.tr("盒子尺寸")
            y_label_text = self.tr("裂缝数量")
            
            # 2. 用代码拼接数学公式和翻译好的文本
            self.fractal_3d_canvas.axes.set_xlabel(rf'$\log_{{10}}$({x_label_text})', fontsize=12, 
                                                fontfamily='Microsoft YaHei')
            self.fractal_3d_canvas.axes.set_ylabel(rf'$\log_{{10}}$({y_label_text})', fontsize=12, 
                                                fontfamily='Microsoft YaHei')
            
            self.fractal_3d_canvas.axes.grid(True, alpha=0.3)
            
            title = self.tr('三维分形维数对比\n实际: D={0:.3f}, $N_0$={1:.1f}, R²={2:.3f}\n理论: D={3:.3f}, $N_0$={4:.1f}').format(
                self.latest_calculated_dim, 10**self.latest_fit_data['coeffs'][1], self.latest_fit_data['r_squared'], self.fractal_dim_spin.value(), self.n0_spin.value()
            )
            self.fractal_3d_canvas.axes.set_title(title, fontsize=11, pad=10, 
                                                fontfamily='Microsoft YaHei')
            
            # 设置标题
            r_squared = fit_data['r_squared']
            calculated_n0 = 10 ** coeffs[1]
            title = self.tr('三维分形维数对比\n实际: D={0:.3f}, $N_0$={1:.1f}, R²={2:.3f}\n理论: D={3:.3f}, $N_0$={4:.1f}').format(
                calculated_dim, 10**coeffs[1], fit_data['r_squared'], expected_dim, expected_n0
            )
            self.fractal_3d_canvas.axes.set_title(title, fontsize=11, pad=10, 
                                                 fontfamily='Microsoft YaHei')
            
            # 添加图例
            self.fractal_3d_canvas.axes.legend(loc='best', fontsize=10)
            
            # 添加详细数据信息
            data_text = self.tr("盒子尺寸 / 实际计数 / 理论计数") + "\n"
            for i, (size, actual, theory) in enumerate(zip(actual_box_sizes, actual_counts, theory_counts)):
                data_text += f"{size:.3f} / {actual} / {theory:.1f}\n"
            
            # 将数据信息放在图的左下角
            self.fractal_3d_canvas.axes.text(0.02, 0.02, data_text, transform=self.fractal_3d_canvas.axes.transAxes,
                                            fontsize=12, verticalalignment='bottom', fontfamily='Microsoft YaHei',
                                            )
        else:
            self.fractal_3d_canvas.axes.text(0.5, 0.5, self.tr('数据不足，无法绘制对比图'), 
                                           transform=self.fractal_3d_canvas.axes.transAxes, 
                                           ha='center', va='center', fontsize=12, 
                                           fontfamily='Microsoft YaHei')
        
        # --- 新增：调用tight_layout()来自动调整布局 ---
        self.fractal_3d_canvas.fig.tight_layout()

        # 刷新画布
        self.fractal_3d_canvas.draw()
    
# 文件: FD-Advanced_3D.py (在 FractalFractureGeneratorGUI 类中)

    def generate_fractures(self):
        """生成分形裂缝（使用统一停止逻辑和计时器）"""
        self.stop_all_processes()
        
        # 步骤1：获取所有参数
        cube_size = self.cube_size_spin.value()
        fractal_dim = self.fractal_dim_spin.value()
        n0 = self.n0_spin.value()
        iterations = self.iterations_spin.value()
        
        # (保留) 获取长短轴参数
        random_aspect_ratio = self.random_aspect_ratio_check.isChecked()
        aspect_ratio_base = self.aspect_ratio_base_spin.value()
        aspect_ratio_variation = self.aspect_ratio_variation_spin.value()
        
        # (新增) 获取新的产状分布参数
        is_isotropic = self.isotropic_check.isChecked()
        mean_inclination = self.mean_inclination_spin.value()
        mean_azimuth = self.mean_azimuth_spin.value()
        kappa = self.kappa_spin.value()
        use_advanced_model = self.advanced_model_check.isChecked()
        
        # 步骤2：准备UI
        self._create_dynamic_tabs(iterations)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.stats_text.setText("正在生成分形裂缝，请稍候...")
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer_label.setText("0.0 s")
        self.timer_label.setVisible(True)
        self.start_time = time.time()
        self.runtime_timer.start()
        num_slices = self.main_num_slices_spin.value() # <--- 从UI获取值

        
        # 步骤3：创建并启动生成线程（传递新的参数）
        self.generation_thread = FractureGenerationThread(
            self.generator, cube_size, fractal_dim, n0, iterations,
            random_aspect_ratio, aspect_ratio_base, aspect_ratio_variation,
            is_isotropic, mean_inclination, mean_azimuth, kappa, 
            use_advanced_model, 
            num_slices,
        )
        
        self.generation_thread.generation_finished.connect(self.on_generation_finished)
        self.generation_thread.generation_failed.connect(self.on_generation_failed)
        self.generation_thread.progress_updated.connect(self.on_generation_progress)
        
        self.generation_thread.start()

    def on_generation_finished(self, fractures_by_level, structured_log):
        """处理生成完成信号，并自动衔接计算"""
        # 检查线程是否在等待期间被用户中止
        if not self.generation_thread:
            return
            
        try:
            self.latest_fractures = self.generator.get_fractures_up_to_level(self.iterations_spin.value())
            self.latest_detailed_stats = structured_log
            
            # ==================== 任务三：集中调用绘图 ====================
            self.update_all_plots_from_data(show_progress=True)
            # ==================== 修改结束 ====================
            
            rendered_text = self._render_generation_log(self.latest_detailed_stats)
            self.stats_text.setText(rendered_text)

            # 自动、无缝地启动计算流程
            self.calculate_fractal_dimension()
            
            # 启用与模型相关的其他功能
            self.export_btn.setEnabled(True)
            self.capture_3d_view_btn.setEnabled(True)

            if self.drilling_study_btn:
                self.drilling_study_btn.setEnabled(True)
            self.capture_section_btn.setEnabled(True)
            
            # --- 新增 ---
            if hasattr(self, 'analyze_trend_btn'):
                self.analyze_trend_btn.setEnabled(True)
            # -----------

            if self.unfolding_analysis_btn:
                self.unfolding_analysis_btn.setEnabled(True)
            
            # 更新截面捕获UI
            cube_size = self.cube_size_spin.value()
            self.coordinate_spin.setRange(0, cube_size)
            self.coordinate_spin.setValue(cube_size / 2)
            self.position_slider.setEnabled(True)
            self.auto_capture_checkbox.setEnabled(True)
            self.auto_calc_fractal_checkbox.setEnabled(True)
            
            if hasattr(self, 'tab_widget') and self.tab_widget is not None:
                max_iteration = self.iterations_spin.value()
                if max_iteration < self.tab_widget.count():
                    self.tab_widget.setCurrentIndex(max_iteration)
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr(f"处理生成结果时发生错误:\n{str(e)}"))
            self.stop_all_processes() # 如果出错，调用统一的停止函数来恢复UI

        # 在 on_generation_finished 方法的末尾添加
        # 刷新表面展开分析标签页（如果已打开）
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, UnfoldingAnalysisWidget):
                widget.update_analysis(self.latest_fractures, self.cube_size_spin.value())
                break

    def update_all_plots_from_data(self, show_progress=True):
        """
        一个新的聚合函数，用于在数据生成后一次性更新所有相关的绘图。
        """
        self.update_displays(show_progress=show_progress)
        
        if self.latest_fractures:
            self.plot_statistics_analysis(show_progress=show_progress)
            self.calculate_and_plot_face_views(show_progress=show_progress)

    def on_generation_failed(self, error_message):
        """处理生成失败信号"""
        QMessageBox.critical(self, self.tr("错误"), self.tr("生成失败: {0}").format(error_message))
        self.stop_all_processes() # 调用统一的停止函数来恢复UI

    def on_generation_progress(self, message):
        """处理生成进度更新信号"""
        self.stats_text.setText(message)
        QApplication.processEvents()
    

    
    # 请用这个新的、完整的函数替换您代码中的同名函数

# 文件: FD-Advanced_3D.py

    def _generate_enhanced_stats(self, rendered_original_stats, calculated_dim, fit_data, levels, expected_dim, expected_n0):
        """生成包含分形维数验证的增强统计信息"""
        enhanced_stats = []
        # --- 核心修改：直接使用渲染好的文本 ---
        enhanced_stats.append(rendered_original_stats)
        
        enhanced_stats.append("")
        enhanced_stats.append("=" * 60)
        enhanced_stats.append(self.tr("分形维数验证计算"))
        enhanced_stats.append("=" * 60)
        
        if calculated_dim is not None and fit_data is not None:
            calculated_n0 = 10 ** fit_data['coeffs'][1]
            
            enhanced_stats.append(self.tr("期望参数:"))
            enhanced_stats.append(self.tr("  期望分形维数 D = {0:.3f}").format(expected_dim))
            enhanced_stats.append(self.tr("  期望分形初值 N0 = {0:.3f}").format(expected_n0))
            enhanced_stats.append("")
            
            enhanced_stats.append(self.tr("计算结果:"))
            enhanced_stats.append(self.tr("  计算分形维数 D = {0:.3f}").format(calculated_dim))
            enhanced_stats.append(self.tr("  计算分形初值 N0 = {0:.3f}").format(calculated_n0))
            enhanced_stats.append(self.tr("  拟合优度 R² = {0:.3f}").format(fit_data['r_squared']))
            enhanced_stats.append("")
            
            dim_error = abs(calculated_dim - expected_dim)
            dim_error_percent = (abs(calculated_dim - expected_dim) / expected_dim * 100) if expected_dim != 0 else float('inf')
            n0_error = abs(calculated_n0 - expected_n0)
            n0_error_percent = (abs(calculated_n0 - expected_n0) / expected_n0 * 100) if expected_n0 != 0 else float('inf')
            
            enhanced_stats.append(self.tr("验证结果:"))
            enhanced_stats.append(self.tr("  分形维数偏差: {0:.3f} ({1:.3f}%)").format(dim_error, dim_error_percent))
            enhanced_stats.append(self.tr("  分形初值偏差: {0:.3f} ({1:.3f}%)").format(n0_error, n0_error_percent))
            
            if dim_error < 0.1 and n0_error_percent < 20: accuracy = self.tr("优秀")
            elif dim_error < 0.2 and n0_error_percent < 30: accuracy = self.tr("良好")
            elif dim_error < 0.3 and n0_error_percent < 50: accuracy = self.tr("一般")
            else: accuracy = self.tr("较差")
            
            enhanced_stats.append(self.tr("  总体精度评估: {0}").format(accuracy))
            enhanced_stats.append("")
            
            enhanced_stats.append(self.tr("详细计算数据:"))
            header_box_size = self.tr('盒子尺寸')
            header_valid_count = self.tr('有效计数')
            enhanced_stats.append(f"  {header_box_size:<12} |    {header_valid_count}")
            enhanced_stats.append("  ------------------------------")
            
            for level in levels:
                enhanced_stats.append(f"  {level['box_size']:>12.6f} {level['valid_count']:>12}")
            
            enhanced_stats.append("")
            enhanced_stats.append(self.tr("拟合方程: log₁₀(N) = {0:.3f}·log₁₀(L) + {1:.3f}").format(fit_data['coeffs'][0], fit_data['coeffs'][1]))
            
        else:
            enhanced_stats.append(self.tr("分形维数计算失败：数据不足或无有效数据点"))
            enhanced_stats.append(self.tr("建议："))
            enhanced_stats.append(self.tr("  1. 增加裂缝数量（提高N0值）"))
            enhanced_stats.append(self.tr("  2. 调整分形维数范围"))
            enhanced_stats.append(self.tr("  3. 增加迭代次数"))
        
        return "\n".join(enhanced_stats)
    
    def calculate_fractal_dimension(self):
        """计算分形维数（作为流程的一部分被调用）"""
        if not self.latest_fractures:
            self.stop_all_processes() # 如果没有裂缝，也应该停止流程
            return
        
        cube_size = self.cube_size_spin.value()
        calc_iterations = self.calc_iterations_spin.value()
        
        # 创建并启动计算线程
        self.calculation_thread = FractalDimensionCalculationThread(
            self.latest_fractures, cube_size, calc_iterations
        )
        
        self.calculation_thread.calculation_finished.connect(self.on_calculation_finished)
        self.calculation_thread.calculation_failed.connect(self.on_calculation_failed)
        self.calculation_thread.progress_updated.connect(self.on_calculation_progress)
        
        # UI状态由外部管理，这里只需启动线程
        self.calculation_thread.start()

    def on_calculation_progress(self, message):
        """计算进度更新"""
        self.stats_text.append(f"\n{message}")
        QApplication.processEvents()

    def on_calculation_finished(self, calculated_dim, fit_data, levels):
        """计算完成处理（流程的最终成功节点）"""
        # 检查线程是否在等待期间被用户中止
        if not self.calculation_thread:
            return

        # --- 核心修改：停止QTimer ---
        if self.runtime_timer.isActive():
            self.runtime_timer.stop()
            # 确保显示的是最终的精确时间
            elapsed_time = time.time() - self.start_time
            self.timer_label.setText(f"{elapsed_time:.1f} s")
        # --- 修改结束 ---

        self.latest_calculated_dim = calculated_dim
        self.latest_fit_data = fit_data
        self.latest_levels = levels

        fractal_dim = self.fractal_dim_spin.value()
        n0 = self.n0_spin.value()
        
        rendered_base_stats = self._render_generation_log(self.latest_detailed_stats)
        enhanced_stats = self._generate_enhanced_stats(
            rendered_base_stats, calculated_dim, fit_data, levels, fractal_dim, n0
        )
        self.stats_text.setText(enhanced_stats)
        
        if self.fractal_3d_canvas:
            self._plot_3d_fractal_comparison_with_data()
        

        
        # 调用统一的停止/恢复函数来结束流程
        self.stop_all_processes()

    def on_calculation_failed(self, error_message):
        """计算失败处理（流程的最终失败节点）"""
        # 检查线程是否在等待期间被用户中止
        if not self.calculation_thread:
            return

        current_text = self.stats_text.toPlainText()
        error_text = f"\n\n{self.tr('分形维数计算失败')}:\n{error_message}"
        self.stats_text.setText(current_text + error_text)
        
        # 调用统一的停止/恢复函数来结束流程
        self.stop_all_processes()

    
# 文件: FD-Advanced_3D.py

    def update_displays(self, show_progress=True):
        """更新所有显示画布（优化版本）"""
        cube_size = self.cube_size_spin.value()
        total_canvases = len(self.canvases)
        
        for i, (max_level, canvas) in enumerate(self.canvases.items()):
            if show_progress and hasattr(self, 'stats_text'):
                # --- 核心修改：翻译进度消息 ---
                progress_msg = self.tr("正在更新显示 {0}/{1}...").format(i + 1, total_canvases)
                self.stats_text.setText(progress_msg)
                QApplication.processEvents()
            
            canvas.axes.clear()
            self.draw_cube_outline(canvas.axes, cube_size)
            fractures = self.generator.get_fractures_up_to_level(max_level)
            
            batch_size = 50
            for batch_start in range(0, len(fractures), batch_size):
                batch_end = min(batch_start + batch_size, len(fractures))
                batch_fractures = fractures[batch_start:batch_end]
                for fracture in batch_fractures:
                    self.draw_fracture(canvas.axes, fracture)
                if batch_end < len(fractures):
                    QApplication.processEvents()
            
            canvas.axes.set_xlim([0, cube_size])
            canvas.axes.set_ylim([0, cube_size])
            canvas.axes.set_zlim([0, cube_size])
            canvas.axes.set_xlabel(self.tr('X轴'), fontfamily='Microsoft YaHei')
            canvas.axes.set_ylabel(self.tr('Y轴'), fontfamily='Microsoft YaHei')
            canvas.axes.set_zlabel(self.tr('Z轴'), fontfamily='Microsoft YaHei')
            
            title = self._generate_title_for_level(max_level, len(fractures))
            canvas.axes.set_title(title, fontfamily='Microsoft YaHei', fontsize=12)
            canvas.axes.set_box_aspect([1,1,1])
            canvas.draw()
            QApplication.processEvents()
    
    def _generate_title_for_level(self, level, fracture_count):
        """为指定层级生成标题"""
        if level == 0:
            return self.tr('初始状态 ({0} 个裂缝面)').format(fracture_count)
        elif level == 1:
            return self.tr('第一次迭代 ({0} 个裂缝面)').format(fracture_count)
        elif level == 2:
            return self.tr('第二次迭代 ({0} 个裂缝面)').format(fracture_count)
        elif level == 3:
            return self.tr('第三次迭代 ({0} 个裂缝面)').format(fracture_count)
        elif level == 4:
            return self.tr('第四次迭代 ({0} 个裂缝面)').format(fracture_count)
        else:
            return self.tr('第{0}次迭代 ({1} 个裂缝面)').format(level, fracture_count)
    
    def draw_cube_outline(self, axes, size):
        """绘制立方体边框"""
        vertices = [
            [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
            [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]
        ]
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for edge in edges:
            points = np.array([vertices[edge[0]], vertices[edge[1]]])
            axes.plot3D(
                points[:, 0], points[:, 1], points[:, 2],
                'k-', alpha=0.6, linewidth=1
            )
    
    def draw_fracture(self, axes, fracture):
        """绘制单个裂缝面"""
        # 使用裁切后的顶点进行显示
        if fracture.clipped_vertices is not None and len(fracture.clipped_vertices) >= 3:
            vertices = fracture.clipped_vertices
        else:
            vertices = fracture.vertices
        
        if len(vertices) < 3:
            return
        
        try:
            # 透明显示模式：绘制裂缝面
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            # 根据迭代层级选择颜色和透明度
            colors = [
                ('royalblue', 0.6),      # 初始状态：蓝色
                ('forestgreen', 0.5),    # 第一次迭代：绿色
                ('orange', 0.5),         # 第二次迭代：橙色
                ('crimson', 0.5),        # 第三次迭代：红色
                ('purple', 0.5),         # 第四次迭代：紫色
                ('gold', 0.5),           # 第五次迭代：金色
            ]
            
            # 获取对应层级的颜色
            if fracture.iteration_level < len(colors):
                color, alpha = colors[fracture.iteration_level]
            else:
                color, alpha = ('gray', 0.5)  # 默认颜色
            
            # 绘制填充面
            collection = Poly3DCollection([vertices], alpha=alpha, facecolor=color, 
                                        linewidths=0.8, edgecolors='darkblue')
            axes.add_collection3d(collection)
            
            # 绘制裂缝边界
            vertices_closed = np.vstack([vertices, vertices[0]])
            edge_color = 'red' if fracture.was_clipped else 'darkblue'
            
            axes.plot(
                vertices_closed[:, 0], 
                vertices_closed[:, 1], 
                vertices_closed[:, 2],
                color=edge_color, linewidth=1.0, alpha=0.9
            )
            
        except Exception as e:
            print(f"绘制裂缝面时出错: {e}")
    
    def toggle_aspect_ratio_controls(self, checked):
        """切换长短轴比例控制显示"""
        self.aspect_ratio_controls.setVisible(not checked)

    def toggle_orientation_mode(self, checked):
        """切换产状生成模式（全随机 vs. 高级模式 vs. 自定义各向异性）"""
        if checked:
            # 如果选中"全随机"，禁用其他两者
            self.anisotropic_controls.setEnabled(False)
            if self.advanced_model_check.isChecked():
                self.advanced_model_check.blockSignals(True)
                self.advanced_model_check.setChecked(False)
                self.advanced_model_check.blockSignals(False)
        else:
            # 如果取消选中"全随机"
            # 只有当"高级模式"也没选中时，才启用自定义参数
            if not self.advanced_model_check.isChecked():
               self.anisotropic_controls.setEnabled(True)

    def toggle_advanced_mode(self, checked):
        """切换高级产状模式"""
        if checked:
            # 如果选中"高级模式"，禁用其他两者
            self.anisotropic_controls.setEnabled(False)
            if self.isotropic_check.isChecked():
                self.isotropic_check.blockSignals(True)
                self.isotropic_check.setChecked(False)
                self.isotropic_check.blockSignals(False)
        else:
            # 如果取消选中"高级模式"
            # 只有当"全随机"也没选中时，才启用自定义参数
            if not self.isotropic_check.isChecked():
                self.anisotropic_controls.setEnabled(True)

    def _sync_spinbox_value(self, target_attr, value):
        """通用的SpinBox同步方法"""
        target_spinbox = getattr(self, target_attr, None)
        if target_spinbox and target_spinbox.value() != value:
            target_spinbox.blockSignals(True)
            target_spinbox.setValue(value)
            target_spinbox.blockSignals(False)

# 文件: FD-Advanced_3D.py

    # 文件: FD-Advanced_3D.py

    # 请用这个完整的函数替换您代码中旧的 clear_display 函数
# 文件: FD-Advanced_3D.py

    def clear_display(self):
        """清除所有显示（最终修复版，增加完整安全检查）"""
        self.stop_all_processes()

        self._create_default_tabs()
        
        self.latest_fractures = []
        self.latest_detailed_stats = []
        self.latest_calculated_dim = None
        self.latest_fit_data = None
        self.latest_levels = None
        self.latest_face_analysis_data = None
        
        if hasattr(self, 'average_fractal_canvas') and self.average_fractal_canvas:
            self.average_fractal_canvas.axes.clear()
            self.average_fractal_canvas.draw()
        if hasattr(self, 'average_data_canvas') and self.average_data_canvas:
            self.average_data_canvas.axes.clear()
            self.average_data_canvas.draw()
        
        self.export_btn.setEnabled(False)
        self.capture_section_btn.setEnabled(False)
        
        # --- 新增 ---
        if hasattr(self, 'analyze_trend_btn'):
            self.analyze_trend_btn.setEnabled(False)
        # -----------

        self.capture_3d_view_btn.setEnabled(False)

        
        # --- 新增：禁用统计图导出按钮 ---
        if hasattr(self, 'export_stats_btn'):
            self.export_stats_btn.setEnabled(False)
        
        if self.drilling_study_btn:
            self.drilling_study_btn.setEnabled(False)
            
        if self.tab_widget:
            for i in range(self.tab_widget.count() - 1, -1, -1):
                widget = self.tab_widget.widget(i)
                if widget and widget.objectName() == "drilling_study_tab_widget":
                    self.tab_widget.removeTab(i)
                    break
                
        if hasattr(self, 'position_slider'):
            self.position_slider.setEnabled(False)
        if hasattr(self, 'auto_capture_checkbox'):
            self.auto_capture_checkbox.setEnabled(False)
            self.auto_capture_checkbox.setChecked(False)
            self.auto_capture_enabled = False
        if hasattr(self, 'auto_calc_fractal_checkbox'):
            self.auto_calc_fractal_checkbox.setEnabled(False)
            self.auto_calc_fractal_checkbox.setChecked(False)
        
        self.clear_drilling_analysis()

                # 在 clear_display 方法中添加
        # 清除表面展开分析标签页的内容（如果已打开）
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, UnfoldingAnalysisWidget):
                widget.clear_results()
                break
        
        self.stats_text.setText(self.tr("请设置参数并点击'生成分形裂缝'按钮开始生成..."))
        if hasattr(self, 'timer_label'):
            self.timer_label.setVisible(False)

        if self.unfolding_analysis_btn:
            self.unfolding_analysis_btn.setEnabled(False)
        # --- 新增结束 ---

# 文件: FD-Advanced_3D.py
# 替换目标: class FractalFractureGeneratorGUI -> def open_unfolding_analyzer(self)

    def open_unfolding_analyzer(self):
        """按需创建或切换到表面展开分析标签页"""
        # 查找是否已存在该标签页
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, UnfoldingAnalysisWidget):
                self.tab_widget.setCurrentIndex(i)
                return

        # ===================== 修正开始 =====================
        # 如果不存在，则创建新的标签页，并将主窗口实例(self)传递给它
        unfolding_widget = UnfoldingAnalysisWidget(main_window=self)
        # ===================== 修正结束 =====================
        
        # 为标签页设置一个唯一的对象名称，以便翻译函数识别
        unfolding_widget.setObjectName("unfolding_analysis_tab_widget")
        
        tab_index = self.tab_widget.addTab(unfolding_widget, self.tr("表面展开分析"))
        self.tab_widget.setCurrentIndex(tab_index)
        
        # 如果当前已有裂缝数据，则立即更新新创建的标签页内容
        if self.latest_fractures:
            unfolding_widget.update_analysis(self.latest_fractures, self.cube_size_spin.value())
    
    def on_section_type_changed(self):
        """截面类型改变时更新标签"""
        section_type = self.section_type_combo.currentText()
        if "YOZ" in section_type:
            self.coordinate_spin.setSuffix(" (X值)")
        elif "XOZ" in section_type:
            self.coordinate_spin.setSuffix(" (Y值)")
        elif "XOY" in section_type:
            self.coordinate_spin.setSuffix(" (Z值)")
        
        # 根据截面类型更新坐标范围
        cube_size = self.cube_size_spin.value()
        if cube_size > 0:
            self.coordinate_spin.setRange(0, cube_size)
            self.update_slider_range()
    
    def on_coordinate_changed(self):
        """坐标值输入框变化时的处理"""
        if not self.slider_updating:
            self.slider_updating = True
            # 将输入框的值转换为滑动条的位置
            cube_size = self.cube_size_spin.value()
            if cube_size > 0:
                relative_position = self.coordinate_spin.value() / cube_size
                slider_position = int(relative_position * 1000)
                self.position_slider.setValue(slider_position)
            self.slider_updating = False
            
            # 如果启用了自动捕获，则自动捕获截面
            if self.auto_capture_enabled and self.latest_fractures:
                self.capture_section()
    
    def on_slider_changed(self):
        """滑动条变化时的处理"""
        if not self.slider_updating:
            self.slider_updating = True
            # 将滑动条的位置转换为绝对坐标值
            cube_size = self.cube_size_spin.value()
            if cube_size > 0:
                relative_position = self.position_slider.value() / 1000.0
                absolute_position = relative_position * cube_size
                self.coordinate_spin.setValue(absolute_position)
            self.slider_updating = False
            
            # 如果启用了自动捕获，则自动捕获截面
            if self.auto_capture_enabled and self.latest_fractures:
                self.capture_section()
    
    def on_auto_capture_changed(self, state):
        """自动捕获选项变化时的处理"""
        self.auto_capture_enabled = state == Qt.Checked
    
    def update_slider_range(self):
        """更新滑动条范围"""
        cube_size = self.cube_size_spin.value()
        if cube_size > 0:
            # 滑动条范围保持0-1000，表示相对位置
            # 实际坐标值通过相对位置计算
            pass
    
    def capture_section(self):
        """捕获截面交线"""
        if not self.latest_fractures:
            QMessageBox.warning(self, self.tr("警告"), self.tr("请先生成空间分形裂缝系统！"))
            return
        if self.section_type_combo.currentIndex() == 3:
            QMessageBox.information(self, self.tr("提示"), self.tr("“三个方向平均”选项仅用于【分析切片趋势】，无法捕获单一截面图像。\n请选择具体的平面类型（YOZ/XOZ/XOY）进行捕获。"))
            return
        
        try:
            section_type = self.section_type_combo.currentIndex()
            coordinate_value = self.coordinate_spin.value()
            cube_size = self.cube_size_spin.value()
            
            # 计算截面与裂缝的交线
            intersection_lines = self.calculate_section_intersections(
                section_type, coordinate_value
            )
            
            # 检查是否勾选了自动计算分形维数
            auto_calc_fractal = self.auto_calc_fractal_checkbox.isChecked()
            
            if auto_calc_fractal:
                # 使用增强版对话框（三列布局）
                # 检查是否已经存在增强版截面结果窗口
                if hasattr(self, 'enhanced_section_result_dialog') and self.enhanced_section_result_dialog is not None and self.enhanced_section_result_dialog.isVisible():
                    # 如果窗口存在且可见，则刷新数据
                    self.enhanced_section_result_dialog.update_data(intersection_lines, section_type, coordinate_value, cube_size)
                    self.enhanced_section_result_dialog.raise_()  # 将窗口提到前台
                    self.enhanced_section_result_dialog.activateWindow()  # 激活窗口
                else:
                    # 如果窗口不存在或不可见，则创建新窗口
                    self.enhanced_section_result_dialog = EnhancedSectionResultDialog(intersection_lines, section_type, coordinate_value, cube_size, self)
                    self.enhanced_section_result_dialog.show()
            else:
                # 使用普通版对话框（单列布局）
                # 检查是否已经存在截面结果窗口
                if self.section_result_dialog is not None and self.section_result_dialog.isVisible():
                    # 如果窗口存在且可见，则刷新数据
                    self.section_result_dialog.update_data(intersection_lines, section_type, coordinate_value, cube_size)
                    self.section_result_dialog.raise_()  # 将窗口提到前台
                    self.section_result_dialog.activateWindow()  # 激活窗口
                else:
                    # 如果窗口不存在或不可见，则创建新窗口
                    self.section_result_dialog = SectionResultDialog(intersection_lines, section_type, coordinate_value, cube_size, self)
                    self.section_result_dialog.show()
            
            # 自动勾选"自动捕获截面"复选框
            if not self.auto_capture_checkbox.isChecked():
                self.auto_capture_checkbox.setChecked(True)
                self.auto_capture_enabled = True
            
        except Exception as e:
            QMessageBox.warning(self, self.tr("错误"), self.tr(f"捕获截面时发生错误：{str(e)}"))
    
    def calculate_section_intersections(self, section_type, coordinate_value):
        """计算截面与裂缝的交线"""
        intersection_lines = []
        tolerance = 1e-6
        
        for fracture in self.latest_fractures:
            # 使用裁切后的顶点，如果没有则使用原始顶点
            if fracture.clipped_vertices is not None and len(fracture.clipped_vertices) >= 3:
                vertices = fracture.clipped_vertices
            else:
                vertices = fracture.vertices
            
            if len(vertices) < 3:
                continue
            
            # 计算多边形与平面的交线
            intersection_points = fracture._calculate_polygon_plane_intersection(
                vertices, section_type, coordinate_value, tolerance
            )
            # --- 修改结束 ---
            
            if len(intersection_points) >= 2:
                # 注意：intersection_points 返回的是 list of np.array, 需要转换为 np.array
                intersection_lines.append({
                    'points': np.array(intersection_points),
                    'fracture_id': fracture.fracture_id,
                    'fracture_level': fracture.iteration_level
                })
        
        return intersection_lines

    
    def on_display_mode_changed(self):
        """显示模式改变时重新绘制（保留透明显示模式）"""
        if self.latest_fractures:
            self.update_displays()
    
    def start_drilling_analysis(self):
        """开始钻孔分析（简化为单次）"""
        if not self.latest_fractures:
            QMessageBox.warning(self, self.tr("警告"), self.tr("请先生成空间分形裂缝系统！"))
            return
        
        if self.drilling_analyzer.random_point is None:
            QMessageBox.warning(self, self.tr("警告"), self.tr("请先生成随机点！"))
            return
        
        try:
            cube_size = self.cube_size_spin.value()
            
            # 直接进行单次分析
            drilling_prob_data = self.drilling_analyzer.analyze_drilling_probability(
                self.latest_fractures, cube_size, self.drilling_analyzer.random_point
            )
            
            # 保存分析数据
            self.drilling_analysis_data = {
                'drilling_prob_data': drilling_prob_data,
            }
            
            # 更新界面信息
            self.update_drilling_analysis_info()
            
            # 绘制钻孔概率研究图表
            self.plot_drilling_probability()
            
            # 更新表格
            self.update_drilling_table()
            
            # 启用相关按钮
            self.clear_analysis_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr(f"钻孔分析时发生错误:\n{str(e)}"))
    

    def update_drilling_analysis_info(self):
        """更新钻孔分析信息显示（简化为单次）"""
        if not self.drilling_analysis_data:
            return
        
        drilling_prob_data = self.drilling_analysis_data['drilling_prob_data']
        random_point = drilling_prob_data['random_point']
        cube_size = self.cube_size_spin.value()
        
        # --- 核心修改：先翻译，后格式化 ---
        
        # 1. 翻译“随机点信息”的模板
        point_pos_text = self.tr("随机点位置:")
        analysis_range_text = self.tr("分析范围:")
        
        # 2. 拼接成最终文本
        random_point_full_text = (
            f"{point_pos_text} ({random_point[0]:.3f}, {random_point[1]:.3f}, {random_point[2]:.3f})\n"
            f"{analysis_range_text} X∈[0, {cube_size:.1f}], Y∈[0, {cube_size:.1f}]"
        )
        self.random_point_label.setText(random_point_full_text)
        
        # 3. 翻译“分析结果”的模板
        analysis_result_text = self.tr("分析结果:")
        prob_intersections = drilling_prob_data['total_intersections']
        intersection_line_text = self.tr("• 垂直线穿过 {0} 个裂缝面").format(prob_intersections)
        
        # 4. 拼接成最终文本
        analysis_result_full_text = f"{analysis_result_text}\n{intersection_line_text}"
        self.analysis_result_label.setText(analysis_result_full_text)
    
    def update_drilling_table(self):
        """更新钻孔分析表格"""
        if not self.drilling_analysis_data or not self.drilling_table:
            return
        
        drilling_prob_data = self.drilling_analysis_data['drilling_prob_data']
        intersected_fractures = drilling_prob_data['intersected_fractures']
        
        # 先按Z坐标排序（数值排序而非字符串排序）
        sorted_intersected_fractures = sorted(intersected_fractures, key=lambda x: x['z_coordinate'])
        
        # 设置表格行数
        self.drilling_table.setRowCount(len(sorted_intersected_fractures))
        
        # 填充表格数据
        for i, intersection_info in enumerate(sorted_intersected_fractures):
            fracture = intersection_info['fracture']
            fracture_id = intersection_info['fracture_id']
            
            # 裂缝ID
            self.drilling_table.setItem(i, 0, QTableWidgetItem(str(fracture_id)))
            
            # Z坐标
            z_coord = intersection_info['z_coordinate']
            self.drilling_table.setItem(i, 1, QTableWidgetItem(f"{z_coord:.3f}"))
            
            # 面积
            area = intersection_info['fracture_area']
            self.drilling_table.setItem(i, 2, QTableWidgetItem(f"{area:.3f}"))
            
            # 长轴
            major_axis = fracture.semi_major_axis
            self.drilling_table.setItem(i, 3, QTableWidgetItem(f"{major_axis:.3f}"))
            
            # 短轴
            minor_axis = fracture.semi_minor_axis
            self.drilling_table.setItem(i, 4, QTableWidgetItem(f"{minor_axis:.3f}"))
            
            # 长短轴比
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
            self.drilling_table.setItem(i, 5, QTableWidgetItem(f"{aspect_ratio:.2f}"))
            
            # 迭代层级
            self.drilling_table.setItem(i, 6, QTableWidgetItem(str(fracture.iteration_level)))
            
            # 倾角（度）
            inclination_deg = np.rad2deg(fracture.inclination)
            self.drilling_table.setItem(i, 7, QTableWidgetItem(f"{inclination_deg:.1f}"))
            
            # 方位角（度）
            azimuth_deg = np.rad2deg(fracture.azimuth)
            self.drilling_table.setItem(i, 8, QTableWidgetItem(f"{azimuth_deg:.1f}"))
    
# 文件: FD-Advanced_3D.py

    # 请用这个完整的函数替换您代码中旧的 clear_drilling_analysis 函数
    def clear_drilling_analysis(self):
        """清除钻孔分析结果（最终修复版，增加完整安全检查）"""
        # 清空画布
        if hasattr(self, 'drilling_probability_canvas') and self.drilling_probability_canvas:
            self.drilling_probability_canvas.axes.clear()
            # --- 核心修改：在这里也使用 self.tr() ---
            initial_text = self.tr('请先生成随机点，然后开始钻孔分析')
            self.drilling_probability_canvas.axes.text(0.5, 0.5, 0.5, initial_text, 
                                                    transform=self.drilling_probability_canvas.axes.transAxes,
                                                    ha='center', va='center', fontsize=14)
            self.drilling_probability_canvas.draw()
        
        # 清空表格
        if hasattr(self, 'drilling_table') and self.drilling_table:
            self.drilling_table.setRowCount(0)
        
        # 重置信息标签 (这些是导致崩溃的关键点)
        if hasattr(self, 'random_point_label') and self.random_point_label:
            self.random_point_label.setText(self.tr("随机点位置: 未生成"))
            
        if hasattr(self, 'analysis_result_label') and self.analysis_result_label:
            self.analysis_result_label.setText(self.tr("分析结果: 未开始分析"))
        
        # 清空分析数据
        self.drilling_analysis_data = None
        # drilling_analyzer 是在 __init__ 中创建的，所以它一定存在，但以防万一
        if self.drilling_analyzer:
            self.drilling_analyzer.random_point = None
        
        # 禁用相关按钮
        if hasattr(self, 'start_analysis_btn') and self.start_analysis_btn:
            self.start_analysis_btn.setEnabled(False)
            
        if hasattr(self, 'clear_analysis_btn') and self.clear_analysis_btn:
            self.clear_analysis_btn.setEnabled(False)
    
    # 请用这个新的、完整的函数替换您代码中的同名函数

    def plot_drilling_probability(self):
        """绘制钻孔概率研究图表"""
        if not self.drilling_analysis_data or not self.drilling_probability_canvas:
            return
        
        drilling_prob_data = self.drilling_analysis_data['drilling_prob_data']
        
        self.drilling_probability_canvas.axes.clear()
        
        cube_size = self.cube_size_spin.value()
        
        self.draw_cube_outline(self.drilling_probability_canvas.axes, cube_size)
        
        fracture_opacity = self.fracture_opacity_spin.value()
        non_intersected_opacity = self.non_intersected_opacity_spin.value()
        show_non_intersected = self.show_non_intersected_check.isChecked()
        
        intersected_fracture_ids = {info['fracture_id'] for info in drilling_prob_data['intersected_fractures']}
        
        if show_non_intersected:
            for i, fracture in enumerate(self.latest_fractures):
                if fracture.fracture_id not in intersected_fracture_ids: # 使用 fracture_id 检查
                    vertices = fracture.clipped_vertices if fracture.clipped_vertices is not None and len(fracture.clipped_vertices) >= 3 else fracture.vertices
                    if len(vertices) >= 3:
                        try:
                            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                            collection = Poly3DCollection([vertices], alpha=non_intersected_opacity, facecolor='lightblue', edgecolor='lightgray', linewidths=0.3)
                            self.drilling_probability_canvas.axes.add_collection3d(collection)
                        except:
                            pass
        
        for intersection_info in drilling_prob_data['intersected_fractures']:
            fracture = intersection_info['fracture']
            vertices = fracture.clipped_vertices if fracture.clipped_vertices is not None and len(fracture.clipped_vertices) >= 3 else fracture.vertices
            if len(vertices) >= 3:
                try:
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    collection = Poly3DCollection([vertices], alpha=fracture_opacity, facecolor='red', edgecolor='darkred', linewidths=1.5)
                    self.drilling_probability_canvas.axes.add_collection3d(collection)
                except:
                    pass
        
        line_start = drilling_prob_data['line_start']
        line_end = drilling_prob_data['line_end']
        
        # --- 核心修改：国际化图例标签 ---
        self.drilling_probability_canvas.axes.plot(
            [line_start[0], line_end[0]], [line_start[1], line_end[1]], [line_start[2], line_end[2]],
            'g-', linewidth=4, label=self.tr('钻孔轨迹')
        )
        
        for i, intersection_info in enumerate(drilling_prob_data['intersected_fractures']):
            point = intersection_info['intersection_point']
            self.drilling_probability_canvas.axes.scatter(
                point[0], point[1], point[2], color='yellow', s=100, edgecolor='black',
                linewidth=2, label=self.tr('交点') if i == 0 else ""
            )
        
        self.drilling_probability_canvas.axes.set_xlim([0, cube_size])
        self.drilling_probability_canvas.axes.set_ylim([0, cube_size])
        self.drilling_probability_canvas.axes.set_zlim([0, cube_size])
        
        self.drilling_probability_canvas.axes.set_xlabel(self.tr('X轴'), fontfamily='Microsoft YaHei')
        self.drilling_probability_canvas.axes.set_ylabel(self.tr('Y轴'), fontfamily='Microsoft YaHei')
        self.drilling_probability_canvas.axes.set_zlabel(self.tr('Z轴'), fontfamily='Microsoft YaHei')
        
        # --- 核心修改：国际化图表标题 ---
        total_intersections = drilling_prob_data['total_intersections']
        title = self.tr('钻孔概率研究 - 穿过 {0} 个裂缝面').format(total_intersections)
        self.drilling_probability_canvas.axes.set_title(title, fontfamily='Microsoft YaHei', fontsize=12)
        
        self.drilling_probability_canvas.axes.legend(loc='upper right')
        self.drilling_probability_canvas.axes.set_box_aspect([1,1,1])
        self.drilling_probability_canvas.draw()
    

    

    

    
    def on_display_option_changed(self):
        """显示选项改变时重新绘制钻孔概率图"""
        if self.drilling_analysis_data:
            self.plot_drilling_probability()
    
    # 替换 export_model 函数
    def export_model(self):
        """导出模型"""
        if len(self.latest_fractures) == 0:
            QMessageBox.warning(self, self.tr("警告"), self.tr("请先生成裂缝再导出！"))
            return
        
        format_text = self.export_format_combo.currentText()
        
        if 'STL' in format_text:
            file_filter = self.tr("STL文件 (*.stl)"); default_ext = ".stl"
        elif 'OBJ' in format_text:
            file_filter = self.tr("OBJ文件 (*.obj)"); default_ext = ".obj"
        elif 'PLY' in format_text:
            file_filter = self.tr("PLY文件 (*.ply)"); default_ext = ".ply"
        else:
            file_filter = self.tr("文本文件 (*.txt)"); default_ext = ".txt"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, self.tr("导出模型文件"), 
            f"elliptical_fracture_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}{default_ext}",
            file_filter
        )
        
        if filename:
            try:
                cube_size = self.cube_size_spin.value()
                if filename.endswith('.stl'): self.fracture_exporter.export_to_stl(self.latest_fractures, cube_size, filename)
                elif filename.endswith('.obj'): self.fracture_exporter.export_to_obj(self.latest_fractures, cube_size, filename)
                elif filename.endswith('.ply'): self.fracture_exporter.export_to_ply(self.latest_fractures, cube_size, filename)
                else: self.fracture_exporter.export_fracture_data(self.latest_fractures, cube_size, filename)
                
                message_template = self.tr(
                    "模型已成功导出到:\n{0}\n\n"
                    "椭圆裂缝数量: {1}\n"
                    "立方体尺寸: {2}\n"
                    "导出格式: {3}"
                )
                final_message = message_template.format(filename, len(self.latest_fractures), cube_size, format_text)
                QMessageBox.information(self, self.tr("导出成功"), final_message)
                
            except Exception as e:
                QMessageBox.critical(self, self.tr("导出错误"), self.tr("导出失败:\n{0}").format(str(e)))
    
    def generate_random_point_for_analysis(self):
        """生成随机点用于钻孔分析"""
        if not self.latest_fractures:
            QMessageBox.warning(self, self.tr("警告"), self.tr("请先生成空间分形裂缝系统！"))
            return
        
        try:
            cube_size = self.cube_size_spin.value()
            
            # 生成随机点
            random_point = self.drilling_analyzer.generate_random_point(cube_size)
            
            # --- 核心修改：先翻译，后格式化 ---
            
            # 1. 翻译模板
            point_pos_text = self.tr("随机点位置:")
            analysis_range_text = self.tr("分析范围:")
            
            # 2. 拼接成最终文本
            full_text = (
                f"{point_pos_text} ({random_point[0]:.3f}, {random_point[1]:.3f}, {random_point[2]:.3f})\n"
                f"{analysis_range_text} X∈[0, {cube_size:.1f}], Y∈[0, {cube_size:.1f}]"
            )
            
            # 3. 更新界面显示
            self.random_point_label.setText(full_text)
            
            # 启用开始分析按钮
            self.start_analysis_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("错误"), self.tr(f"生成随机点时发生错误:\n{str(e)}"))

# 文件: FD-Advanced_3D.py
# 替换 FractalFractureGeneratorGUI.closeEvent 方法

    def closeEvent(self, event):
        """主窗口关闭事件处理 - 确保所有线程正确停止"""
        print("主窗口关闭事件触发，正在清理后台线程...")
        
        # --- 核心修复：安全关闭常驻的队列监听线程 ---
        if hasattr(self, 'queue_listener') and self.queue_listener:
            self.queue_listener.stop()
        if hasattr(self, 'listener_thread') and self.listener_thread:
            self.listener_thread.quit()
            if not self.listener_thread.wait(3000): # 等待3秒
                print("警告：队列监听线程未能优雅退出，可能被强制终止。")
        
        # 安全地停止所有可能的计算线程
        if self.generation_thread and self.generation_thread.isRunning():
            self.generation_thread.quit()
            self.generation_thread.wait(2000)
        
        if self.calculation_thread and self.calculation_thread.isRunning():
            self.calculation_thread.quit()
            self.calculation_thread.wait(2000)
            
        if self.study_2d3d_thread and self.study_2d3d_thread.isRunning():
            self.study_2d3d_thread.stop_calculation()
            self.study_2d3d_thread.wait(5000)

        # 关闭多进程管理器
        if hasattr(self, 'mp_manager'):
            self.mp_manager.shutdown()

        print("清理完成，程序退出。")
        event.accept()
 
# 文件: FD-Advanced_3D.py
# 在 FractalFractureGeneratorGUI 类中添加这个新函数

    def _render_generation_log(self, structured_log):
        """将结构化的日志数据渲染为可翻译的文本。"""
        rendered_lines = []
        for key, params in structured_log:
            line = ""
            if key == 'header':
                line = f"===== {self.tr(params['title'])} ====="
            elif key == 'params_header':
                line = self.tr("输入参数:")
            elif key == 'param':
                # 注意：参数名称也需要翻译
                param_name_tr = self.tr(params['name'])
                line = f"  {param_name_tr} = {params['value']}"
            elif key == 'stats_initial':
                line = self.tr("初始状态: {count} 个裂缝面").format(**params)
            elif key == 'stats_iteration':
                line = self.tr("第{level}次迭代补充: {count} 个裂缝面").format(**params)
            elif key == 'stats_total':
                line = self.tr("总计: {count} 个裂缝面").format(**params)
            elif key == 'log_start_generation':
                line = self.tr("开始生成分形裂缝: L={L}, D={D}, N0={N0}").format(**params)
            elif key == 'log_initial_state':
                line = self.tr("初始状态 (L={L:.6f}):").format(**params)
            elif key == 'log_initial_theory':
                line = self.tr("  理论数量: {theory_count:.6f}, 概率舍入后: {actual_count}").format(**params)
            elif key == 'log_success_generation':
                line = self.tr("  成功生成: {count} 个裂缝面\n").format(**params)
            elif key == 'log_iteration_state':
                line = self.tr("第{iteration}次迭代 (L={L:.6f}):").format(**params)
            elif key == 'log_iteration_theory':
                line = self.tr("  理论数量: {theory_count:.6f}, 概率舍入后: {actual_count}").format(**params)
            elif key == 'log_iteration_supplement':
                line = self.tr("  继承数量: {inherited_count}, 需要补充: {needed}").format(**params)
            elif key == 'log_success_generation_supplement':
                line = self.tr("  成功生成: {count} 个补充裂缝面\n").format(**params)
            elif key == 'separator':
                line = "" # 空行
            
            if line: # 避免添加空的 'None'
                rendered_lines.append(line)
        
        return "\n".join(rendered_lines)


# --- START OF FILE FD-Advanced_3D.py ---
# ... (在 FractalFractureGeneratorGUI 类中) ...

    def _update_plot_translations(self):
        """
        轻量级函数：仅更新所有现有图表的文本元素（标题、坐标轴等），不重绘数据。
        """
        if not self.latest_fractures:
            return

        # 更新主显示区的标题和坐标轴
        for level, canvas in self.canvases.items():
            fracture_count = len(self.generator.get_fractures_up_to_level(level))
            title = self._generate_title_for_level(level, fracture_count)
            canvas.axes.set_title(title, fontfamily='Microsoft YaHei', fontsize=12)
            canvas.axes.set_xlabel(self.tr('X轴'), fontfamily='Microsoft YaHei')
            canvas.axes.set_ylabel(self.tr('Y轴'), fontfamily='Microsoft YaHei')
            canvas.axes.set_zlabel(self.tr('Z轴'), fontfamily='Microsoft YaHei')
            canvas.draw()

        # 更新统计分析标签页的图表文本
        if self.inclination_canvas:
            self._plot_angle_distribution(self.inclination_canvas, [np.rad2deg(f.inclination) for f in self.latest_fractures], self.tr("倾角"), 0, 180)
            self.inclination_canvas.draw()
        if self.azimuth_canvas:
            self._plot_angle_distribution(self.azimuth_canvas, [np.rad2deg(f.azimuth) for f in self.latest_fractures], self.tr("方位角"), 0, 360)
            self.azimuth_canvas.draw()
        if self.fractal_3d_canvas:
            if self.latest_calculated_dim is not None:
                self._plot_3d_fractal_comparison_with_data()
            else:
                self._plot_3d_fractal_comparison()
            self.fractal_3d_canvas.draw()
            
        # 更新投影面分析标签页的图表文本 (调用使用缓存的快速绘图函数)
        self.plot_face_views()

        # 更新钻孔分析图的文本
        if self.drilling_analysis_data:
            self.plot_drilling_probability()
            self.update_drilling_analysis_info()
# 文件: FD-Advanced_3D.py

# 文件: FD-Advanced_3D.py

    def retranslate_ui(self):
        """核心函数：当语言切换时，更新所有UI元素的文本。"""
        # ... (所有控件的翻译部分，从 self.setWindowTitle 到 self.fractal_3d_label.setText，保持不变) ...
        self.setWindowTitle(self.tr('基于分形维数理论的空间椭圆裂缝生成器 - 增强版（含钻孔分析）'))
        if hasattr(self, 'settings_menu') and self.settings_menu:
            self.settings_menu.setTitle(self.tr('&设置'))
            self.language_menu.setTitle(self.tr('&语言'))
            self.action_zh.setText(self.tr('中文'))
            self.action_en.setText(self.tr('English'))
        self.fractal_group.setTitle(self.tr('分形维数参数'))
        self.cube_size_label.setText(self.tr('立方体边长 L:'))
        self.cube_size_spin.setSuffix(self.tr(' 单位'))
        self.fractal_dim_label.setText(self.tr('分形维数 D:'))
        self.n0_label.setText(self.tr('分形初值 N0:'))
        self.iterations_label.setText(self.tr('生成迭代次数:'))
        self.calc_iterations_label.setText(self.tr('计算迭代次数:'))
        self.main_num_slices_label.setText(self.tr('分析切片数/单方向:')) # <--- 新增翻译

        self.ellipse_group.setTitle(self.tr('裂缝形态参数'))
        self.random_aspect_ratio_check.setText(self.tr('随机长短轴比例'))
        self.aspect_ratio_base_label.setText(self.tr('基础值:'))
        self.aspect_ratio_variation_label.setText(self.tr('波动值:'))

        # (新增) 产状分布翻译
        self.orientation_group.setTitle(self.tr('产状分布'))
        self.isotropic_check.setText(self.tr('各向同性 / 全随机'))
        self.advanced_model_check.setText(self.tr('Advanced model'))
        self.mean_inclination_label.setText(self.tr('平均倾角:'))
        self.mean_inclination_spin.setSuffix(self.tr(' 度'))
        self.mean_azimuth_label.setText(self.tr('平均方位角:'))
        self.mean_azimuth_spin.setSuffix(self.tr(' 度'))
        self.kappa_label.setText(self.tr('集中度 κ:'))
        self.kappa_spin.setToolTip(self.tr('κ=0 表示完全随机（各向同性）。值越大，方向越集中。'))

        self.section_capture_group.setTitle(self.tr('截面捕获'))
        self.section_type_label.setText(self.tr('截面类型:'))
        current_section_index = self.section_type_combo.currentIndex()
        self.section_type_combo.blockSignals(True)
        self.section_type_combo.clear()
        self.section_type_combo.addItem(self.tr('YOZ平面 (X=常数)'))
        self.section_type_combo.addItem(self.tr('XOZ平面 (Y=常数)'))
        self.section_type_combo.addItem(self.tr('XOY平面 (Z=常数)'))
        self.section_type_combo.addItem(self.tr('三个方向平均 (用于趋势分析)'))

        self.section_type_combo.setCurrentIndex(current_section_index if current_section_index != -1 else 0)
        self.section_type_combo.blockSignals(False)
        self.coordinate_label.setText(self.tr('坐标值:'))
        self.position_slider_label.setText(self.tr('位置滑动:'))
        self.auto_capture_checkbox.setText(self.tr('自动捕获截面'))
        self.auto_calc_fractal_checkbox.setText(self.tr('自动计算分形维数'))
        self.capture_section_btn.setText(self.tr('捕获截面'))
        if hasattr(self, 'analyze_trend_btn'):
            self.analyze_trend_btn.setText(self.tr('分析切片趋势'))

        self.export_options_group.setTitle(self.tr('模型导出'))
        self.export_format_label.setText(self.tr('格式:'))
        current_export_index = self.export_format_combo.currentIndex()
        self.export_format_combo.blockSignals(True)
        self.export_format_combo.clear()
        self.export_format_combo.addItem(self.tr('STL格式 (.stl)'))
        self.export_format_combo.addItem(self.tr('OBJ格式 (.obj)'))
        self.export_format_combo.addItem(self.tr('PLY格式 (.ply)'))
        self.export_format_combo.addItem(self.tr('数据文本 (.txt)'))
        self.export_format_combo.setCurrentIndex(current_export_index if current_export_index != -1 else 0)
        self.export_format_combo.blockSignals(False)
        self.export_btn.setText(self.tr('导出模型'))
        self.generate_btn.setText(self.tr('生成并分析模型')) # 原文本: '生成分形裂缝'
        self.stop_btn.setText(self.tr('停止'))
        if hasattr(self, 'capture_3d_view_btn'): # 确保按钮已创建
            self.capture_3d_view_btn.setText(self.tr('捕获三维视图'))

        self.clear_btn.setText(self.tr('清除显示'))
        self.relation_2d3d_btn.setText(self.tr('2D-3D关系研究'))
        self.drilling_study_btn.setText(self.tr('钻孔模拟研究'))
        self.stats_group.setTitle(self.tr('统计信息'))
        if hasattr(self, 'tab_widget') and self.tab_widget:
            for i in range(self.tab_widget.count()):
                widget = self.tab_widget.widget(i)
                if not widget: continue
                obj_name = widget.objectName()
                if obj_name.startswith("iteration_tab_canvas_"):
                    try:
                        level = int(obj_name.split('_')[-1])
                        self.tab_widget.setTabText(i, self._generate_title_for_level(level, -1).split('(')[0].strip())
                    except (ValueError, IndexError): pass
                elif obj_name == "statistics_analysis_tab_widget": self.tab_widget.setTabText(i, self.tr("统计分析"))
                elif obj_name == "study_2d3d_tab_widget": self.tab_widget.setTabText(i, self.tr("2D-3D关系研究"))
                elif obj_name == "drilling_study_tab_widget": self.tab_widget.setTabText(i, self.tr("钻孔概率研究"))
                elif obj_name == "projection_analysis_tab_widget": self.tab_widget.setTabText(i, self.tr("{0}个投影面轨迹分析").format(3 * self.main_num_slices_spin.value()))
                # 在 retranslate_ui 方法的 elif 链中添加
                elif obj_name == "unfolding_analysis_tab_widget":
                    self.tab_widget.setTabText(i, self.tr("表面展开分析"))
        if hasattr(self, 'study_title_label'): 
            self.study_title_label.setText(self.tr("2D-3D关系研究参数"))
            self.study_params_group.setTitle(self.tr("参数设置"))
            self.study_cube_lengths_label.setText(self.tr("立方体长度(L):"))
            self.study_cube_lengths_input.setPlaceholderText(self.tr("例如: 10 20 30 或 (10, 100, 10)"))
            self.study_fractal_dims_label.setText(self.tr("分形维数(D):"))
            self.study_fractal_dims_input.setPlaceholderText(self.tr("例如: 2.5 2.7 或 (2.0, 3.0, 0.1)"))
            self.study_n0_values_label.setText(self.tr("分形初值(N0):"))
            self.study_n0_values_input.setPlaceholderText(self.tr("例如: 100 200 或 (100, 500, 50)"))
            self.study_gen_iter_label.setText(self.tr("生成迭代次数:"))
            self.study_calc_iter_label.setText(self.tr("计算迭代次数:"))
            self.study_repeat_count_label.setText(self.tr("每组重复次数:"))
            self.study_num_slices_label.setText(self.tr("分析切片数/单方向:")) # <--- 新增翻译

            self.study_parallel_group.setTitle(self.tr("并行计算设置"))
            self.study_parallel_mode_label.setText(self.tr("计算模式:"))
            self.study_enable_parallel_checkbox.setText(self.tr("启用并行计算"))

            self.study_thread_count_label.setText(self.tr("并行进程数:"))
            cpu_count = os.cpu_count() or 4
            self.study_thread_count_spin.setToolTip(self.tr("建议设置为物理核心数，当前系统有{0}个逻辑核心，物理核心通常为{1}个").format(cpu_count, cpu_count//2))
            
            self.study_calc_profile_label.setText(self.tr("切片趋势分析:"))
            self.study_calc_profile_checkbox.setText(self.tr("记录每个切片位置的D/N0数据 (勾选将会增加DB文件体积)"))

            self.study_center_only_label.setText(self.tr("中心域优化:"))
            self.study_center_only_checkbox.setText(self.tr("仅计算0.4-0.6L范围 (剔除边界效应，提升计算速度)"))


            self.study_param_estimate_btn.setText(self.tr("参数预估计算"))
            self.study_start_calc_btn.setText(self.tr("开始计算"))
            self.study_stop_calc_btn.setText(self.tr("停止计算"))
            self.study_clear_btn.setText(self.tr("清除结果"))
            self.study_load_backup_btn.setText(self.tr("加载备份文件"))
            if not (hasattr(self, 'study_2d3d_thread') and self.study_2d3d_thread and self.study_2d3d_thread.isRunning()):
                self.study_status_label.setText(self.tr("请设置参数后点击开始计算"))
            self.study_help_group.setTitle(self.tr("功能说明"))
            self.study_help_text.setText(self.tr("""2D-3D关系研究功能说明：

1. 参数设置：
   - 立方体长度/分形维数/分形初值支持三种格式:
     * 单个值: 10
     * 空格分隔: 10 20 30
     * 序列格式: (起点, 终点, 步长)
       例如: (2.0, 3.0, 0.1) 表示从2.0到3.0，
       以0.1为步长进行取值。
       注意：起点和终点都会被包含在序列中（如果可达）。

2. 计算过程：
   - 对所有参数组合进行批量计算
   - 比较理论值与实际计算值
   - 分析2D和3D分形维数的关系

3. 结果显示：
   - 详细的数据表格
   - 可复制和保存结果"""))

        if hasattr(self, 'study_results_title_label'):
            self.study_results_title_label.setText(self.tr("计算结果"))
            self.study_display_mode_label.setText(self.tr("显示模式:"))
            current_display_index = self.study_display_mode_combo.currentIndex()
            self.study_display_mode_combo.blockSignals(True)
            self.study_display_mode_combo.clear()
            self.study_display_mode_combo.addItem(self.tr("统计汇总(平均值±标准差)"))
            self.study_display_mode_combo.addItem(self.tr("原始数据详情"))
            self.study_display_mode_combo.setCurrentIndex(current_display_index if current_display_index != -1 else 0)
            self.study_display_mode_combo.blockSignals(False)
            self.study_copy_btn.setText(self.tr("复制表格内容"))
            self.study_save_btn.setText(self.tr("保存为CSV"))
            self.study_export_raw_btn.setText(self.tr("导出原始数据"))
            self.update_2d3d_results_display()
        if hasattr(self, 'drilling_title_label'):
            self.drilling_title_label.setText(self.tr("钻孔分析参数"))
            self.drilling_display_options_group.setTitle(self.tr("显示选项"))
            self.show_non_intersected_check.setText(self.tr("显示未穿过的裂缝面"))
            self.fracture_opacity_label.setText(self.tr("穿过裂缝透明度:"))
            self.non_intersected_opacity_label.setText(self.tr("未穿过裂缝透明度:"))
            self.drilling_point_info_group.setTitle(self.tr("随机点信息"))
            if self.drilling_analyzer.random_point is None: self.random_point_label.setText(self.tr("随机点位置: 未生成"))
            self.drilling_result_info_group.setTitle(self.tr("分析结果"))
            if self.drilling_analysis_data is None: self.analysis_result_label.setText(self.tr("分析结果: 未开始分析"))
            self.generate_random_point_btn.setText(self.tr("生成随机点"))
            self.start_analysis_btn.setText(self.tr("开始钻孔分析"))
            self.clear_analysis_btn.setText(self.tr("清除分析结果"))
            self.drilling_table_label.setText(self.tr('钻孔穿过的裂缝详细信息'))
            headers = [self.tr('裂缝ID'), self.tr('Z坐标'), self.tr('面积'), self.tr('长轴'), self.tr('短轴'), self.tr('长短轴比'), self.tr('迭代层级'), self.tr('倾角(度)'), self.tr('方位角(度)')]
            self.drilling_table.setHorizontalHeaderLabels(headers)
            if not self.drilling_analysis_data:
                self.drilling_probability_canvas.axes.clear()
                self.drilling_probability_canvas.axes.text(0.5, 0.5, 0.5, self.tr('请先生成随机点，然后开始钻孔分析'), transform=self.drilling_probability_canvas.axes.transAxes, ha='center', va='center', fontsize=14)
                self.drilling_probability_canvas.draw()
        if hasattr(self, 'proj_avg_all_label'):
            num_slices = self.main_num_slices_spin.value()
            self.proj_avg_yoz_label.setText(self.tr("YOZ方向平均数据 (X方向{0}个面平均)").format(num_slices))
            self.proj_avg_xoz_label.setText(self.tr("XOZ方向平均数据 (Y方向{0}个面平均)").format(num_slices))
            self.proj_avg_xoy_label.setText(self.tr("XOY方向平均数据 (Z方向{0}个面平均)").format(num_slices))
        if hasattr(self, 'inclination_label'):
            self.inclination_label.setText(self.tr('倾角分布'))
            self.azimuth_label.setText(self.tr('方位角分布'))
            self.fractal_3d_label.setText(self.tr('三维分形维数拟合对比'))

        if hasattr(self, 'stats_text'):
            if hasattr(self, 'latest_calculated_dim') and self.latest_calculated_dim is not None:
                rendered_base_stats = self._render_generation_log(self.latest_detailed_stats)
                enhanced_stats_text = self._generate_enhanced_stats(
                    rendered_base_stats, self.latest_calculated_dim, self.latest_fit_data,
                    self.latest_levels, self.fractal_dim_spin.value(), self.n0_spin.value()
                )
                self.stats_text.setText(enhanced_stats_text)
            elif hasattr(self, 'latest_detailed_stats') and self.latest_detailed_stats:
                rendered_text = self._render_generation_log(self.latest_detailed_stats)
                self.stats_text.setText(rendered_text)
            else:
                self.stats_text.setText(self.tr("请设置参数并点击'生成分形裂缝'按钮开始生成..."))

        self._update_plot_translations()

        if hasattr(self, 'enhanced_section_result_dialog') and self.enhanced_section_result_dialog and self.enhanced_section_result_dialog.isVisible():
            self.enhanced_section_result_dialog.retranslate_ui()
            
        if hasattr(self, 'section_result_dialog') and self.section_result_dialog and self.section_result_dialog.isVisible():
            self.section_result_dialog.retranslate_ui()
        
        # --- 新增：翻译统计分析标签页的导出按钮 ---
        if hasattr(self, 'export_stats_btn'):
            self.export_stats_btn.setText(self.tr('一键导出所有统计图'))

        if self.unfolding_analysis_btn:
            self.unfolding_analysis_btn.setText(self.tr('表面展开与分形分析'))

        print("UI retranslated.")



# 完整替换这个类
class EnhancedSectionResultDialog(QDialog):
    """增强的截面结果显示对话框 - 三列布局"""
    
    def __init__(self, intersection_lines, section_type, coordinate_value, cube_size, parent=None):
        super().__init__(parent)
        self.intersection_lines = intersection_lines
        self.section_type = section_type
        self.coordinate_value = coordinate_value
        self.cube_size = cube_size
        self.parent = parent
        self.fractal_dim = None
        self.fit_data = None
        self.levels = None
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setGeometry(100, 100, 1500, 800)
        layout = QVBoxLayout()
        columns_layout = QHBoxLayout()
        
        self.create_intersection_column(columns_layout)
        self.create_fractal_column(columns_layout)
        self.create_stats_column(columns_layout)
        
        layout.addLayout(columns_layout)
        
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton()
        self.save_btn.clicked.connect(self.save_image)
        save_layout.addWidget(self.save_btn)
        save_layout.addStretch()
        layout.addLayout(save_layout)
        
        self.setLayout(layout)
        
        # --- 核心修复：在所有控件创建后，显示前，强制刷新一次翻译 ---
        self.retranslate_ui() 
        
        # 在翻译刷新后再进行计算和绘图
        self.calculate_and_plot()
    
    def create_intersection_column(self, parent_layout):
        self.intersection_group = QGroupBox() 
        intersection_layout = QVBoxLayout()
        self.intersection_figure = Figure(figsize=(6, 6))
        self.intersection_canvas = FigureCanvas(self.intersection_figure)
        intersection_layout.addWidget(self.intersection_canvas)
        self.intersection_group.setLayout(intersection_layout)
        parent_layout.addWidget(self.intersection_group, 2)

    def create_fractal_column(self, parent_layout):
        self.fractal_group = QGroupBox()
        fractal_layout = QVBoxLayout()
        self.fractal_figure = Figure(figsize=(6, 6))
        self.fractal_canvas = SquareAspectCanvas(self.fractal_figure)
        fractal_layout.addWidget(self.fractal_canvas)
        self.fractal_group.setLayout(fractal_layout)
        parent_layout.addWidget(self.fractal_group, 2)
    
    def create_stats_column(self, parent_layout):
        self.stats_group = QGroupBox()
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        font = QFont(); font.setPointSize(12)
        self.stats_text.setFont(font)
        stats_layout.addWidget(self.stats_text)
        self.stats_group.setLayout(stats_layout)
        parent_layout.addWidget(self.stats_group, 1)
        
    def calculate_and_plot(self):
        self.plot_intersection_lines()
        self.calculate_fractal_dimension()
        self.plot_fractal_dimension()
        self.display_statistics()
    
    def plot_intersection_lines(self):
        self.intersection_figure.clear()
        ax = self.intersection_figure.add_subplot(111)
        if self.section_type == 0: x_coord, y_coord = 1, 2
        elif self.section_type == 1: x_coord, y_coord = 0, 2
        else: x_coord, y_coord = 0, 1
        cube_boundary = np.array([[0, 0], [self.cube_size, 0], [self.cube_size, self.cube_size], [0, self.cube_size], [0, 0]])
        ax.plot(cube_boundary[:, 0], cube_boundary[:, 1], 'k--', linewidth=1, alpha=0.5)
        if len(self.intersection_lines) > 0:
            for line_data in self.intersection_lines:
                points = line_data['points']
                if len(points) >= 2:
                    x_coords = points[:, x_coord]
                    y_coords = points[:, y_coord]
                    ax.plot(x_coords, y_coords, color='blue', linewidth=2, alpha=0.8)
        ax.set_aspect('equal')
        ax.set_xlim(0, self.cube_size)
        ax.set_ylim(0, self.cube_size)
        self.intersection_figure.tight_layout()
        self.intersection_canvas.draw()
    
    def calculate_fractal_dimension(self):
        """计算截面轨迹的分形维数"""
        if len(self.intersection_lines) == 0:
            self.fractal_dim, self.fit_data, self.levels = None, None, None
            return

        line_segments = []
        for line_data in self.intersection_lines:
            points = line_data['points']
            if len(points) >= 2:
                # 确定正确的坐标轴索引
                if self.section_type == 0: x_coord, y_coord = 1, 2
                elif self.section_type == 1: x_coord, y_coord = 0, 2
                else: x_coord, y_coord = 0, 1
                
                # 提取2D坐标
                line_coords = [(points[i, x_coord], points[i, y_coord]) for i in range(len(points))]
                line_segments.append(line_coords)

        if self.parent and hasattr(self.parent, 'fractal_2d_calculator'):
            # ===================== 修正开始 =====================
            # 1. 从主窗口获取正确的“生成迭代次数”
            gen_iterations = self.parent.iterations_spin.value()
            
            # 2. 将获取到的值传递给正确的 `generation_iterations` 参数
            temp_calculator = FractalDimension2DCalculator(generation_iterations=gen_iterations)
            # ===================== 修正结束 =====================
            
            # 定义截面名称用于缓存
            section_names = [self.tr("YOZ平面"), self.tr("XOZ平面"), self.tr("XOY平面")]
            face_name_template = self.tr("{0} (坐标={1:.2f})")
            face_name = face_name_template.format(section_names[self.section_type], self.coordinate_value)
            
            # --- 核心修复：使用正确的参数 (trajectories, width, height) ---
            self.fractal_dim, self.fit_data, self.levels = temp_calculator.calculate_fractal_dimension(
                line_segments, self.cube_size, self.cube_size
            )
        else:
            self.fractal_dim, self.fit_data, self.levels = None, None, None
    
    def plot_fractal_dimension(self):
        self.fractal_figure.clear()
        ax = self.fractal_figure.add_subplot(111)
        if self.fractal_dim is None or self.fit_data is None:
            ax.text(0.5, 0.5, self.tr('数据不足\n无法计算分形维数'), transform=ax.transAxes, ha='center', va='center', fontsize=12, fontfamily='Microsoft YaHei', bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
            ax.set_title(self.tr('分形维数拟合曲线'), fontfamily='Microsoft YaHei')
            x_label_text = self.tr("盒子尺寸")
            y_label_text = self.tr("裂缝数量")
            ax.set_xlabel(rf'$\log_{{10}}$({x_label_text})', fontfamily='Microsoft YaHei')
            ax.set_ylabel(rf'$\log_{{10}}$({y_label_text})', fontfamily='Microsoft YaHei')
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect('equal')
        else:
            log_sizes = self.fit_data['log_sizes']; log_counts = self.fit_data['log_counts']
            coeffs = self.fit_data['coeffs']; r_squared = self.fit_data['r_squared']
            x_min, x_max = np.min(log_sizes), np.max(log_sizes); y_min, y_max = np.min(log_counts), np.max(log_counts)
            x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 0.5; y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.5
            x_range = x_max - x_min + 2 * x_margin; y_range = y_max - y_min + 2 * y_margin
            max_range = max(x_range, y_range); x_center = (x_min + x_max) / 2; y_center = (y_min + y_max) / 2
            half_range = max_range / 2; ax.set_xlim(x_center - half_range, x_center + half_range); ax.set_ylim(y_center - half_range, y_center + half_range)
            ax.scatter(log_sizes, log_counts, color='red', s=60, alpha=0.8, zorder=3)
            x_fit = np.linspace(x_center - half_range, x_center + half_range, 100)
            y_fit = np.polyval(coeffs, x_fit); ax.plot(x_fit, y_fit, 'b--', linewidth=2, zorder=2)
            x_label_text = self.tr("盒子尺寸"); y_label_text = self.tr("裂缝数量")
            ax.set_xlabel(rf'$\log_{{10}}$({x_label_text})', fontsize=10, fontfamily='Microsoft YaHei')
            ax.set_ylabel(rf'$\log_{{10}}$({y_label_text})', fontsize=10, fontfamily='Microsoft YaHei')
            ax.grid(True, alpha=0.3); ax.set_aspect('equal', adjustable='box')
            calculated_n0 = 10 ** coeffs[1]
            title_template = self.tr('分形维数拟合曲线\nD = {0:.3f}, $N_0$ = {1:.1f}')
            title = title_template.format(self.fractal_dim, calculated_n0)
            if r_squared > 0: title += f', R² = {r_squared:.3f}'
            ax.set_title(title, fontsize=11, pad=10, fontfamily='Microsoft YaHei')
        self.fractal_figure.tight_layout()
        self.fractal_canvas.draw()
    
    # 在 EnhancedSectionResultDialog 类中，用这个新版本替换整个 display_statistics 函数

    def display_statistics(self):
        """显示分形维数统计参数"""
        self.stats_text.clear()
        
        section_names = [self.tr("YOZ平面"), self.tr("XOZ平面"), self.tr("XOY平面")]
        section_name = section_names[self.section_type]
        
        info_text = self.tr("截面信息:") + "\n"
        info_text += self.tr("截面类型: {0}").format(section_name) + "\n"
        info_text += self.tr("坐标值: {0:.2f}").format(self.coordinate_value) + "\n"
        info_text += self.tr("立方体尺寸: {0:.2f}").format(self.cube_size) + "\n"
        info_text += self.tr("交线数量: {0}").format(len(self.intersection_lines)) + "\n\n"
        
        if self.fractal_dim is not None and self.fit_data is not None:
            calculated_n0 = 10 ** self.fit_data['coeffs'][1]
            r_squared = self.fit_data['r_squared']
            
            info_text += self.tr("分形维数计算结果:") + "\n"
            info_text += self.tr("分形维数 D = {0:.4f}").format(self.fractal_dim) + "\n"
            info_text += self.tr("分形初值 N0 = {0:.4f}").format(calculated_n0) + "\n"
            info_text += self.tr("拟合优度 R² = {0:.4f}").format(r_squared) + "\n\n"
            
            if self.levels:
                info_text += self.tr("各层级数据:") + "\n"
                
                # ========== 核心修正 ==========
                # 将 tr() 调用从 f-string 中分离出来，确保翻译工具可以识别
                header_level = self.tr("层级")
                header_box_size = self.tr("盒子尺寸")
                header_valid_count = self.tr("有效计数")
                info_text += f"{header_level}\t{header_box_size}\t{header_valid_count}\n"
                # ========== 修正结束 ==========

                for i, level in enumerate(self.levels):
                    info_text += f"{i}\t{level['box_size']:.3f}\t{level['valid_count']}\n"
        else:
            info_text += self.tr("分形维数计算失败或数据不足") + "\n"
            
        self.stats_text.setText(info_text)
    
    def update_data(self, intersection_lines, section_type, coordinate_value, cube_size):
        self.intersection_lines = intersection_lines
        self.section_type = section_type
        self.coordinate_value = coordinate_value
        self.cube_size = cube_size
        self.calculate_and_plot()
    
    def save_image(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(self, self.tr("保存截面图片"), "", self.tr("PNG图片 (*.png);;JPG图片 (*.jpg);;PDF文档 (*.pdf)"))
            if filename:
                fig = Figure(figsize=(18, 6))
                ax1 = fig.add_subplot(131)
                self.intersection_figure.savefig('temp_intersection.png', dpi=300, bbox_inches='tight'); img1 = plt.imread('temp_intersection.png')
                ax1.imshow(img1); ax1.set_title(self.tr('截面交线图')); ax1.axis('off')
                ax2 = fig.add_subplot(132)
                self.fractal_figure.savefig('temp_fractal.png', dpi=300, bbox_inches='tight'); img2 = plt.imread('temp_fractal.png')
                ax2.imshow(img2); ax2.set_title(self.tr('分形维数拟合曲线图')); ax2.axis('off')
                ax3 = fig.add_subplot(133); ax3.axis('off')
                stats_text = self.stats_text.toPlainText()
                ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=8, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                fig.tight_layout(); fig.savefig(filename, dpi=300, bbox_inches='tight'); plt.close(fig)
                import os
                if os.path.exists('temp_intersection.png'): os.remove('temp_intersection.png')
                if os.path.exists('temp_fractal.png'): os.remove('temp_fractal.png')
                QMessageBox.information(self, self.tr("成功"), self.tr("图片已保存到：{0}").format(filename))
        except Exception as e:
            QMessageBox.critical(self, self.tr("保存失败"), self.tr("保存失败：{0}").format(str(e)))

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("截面交线结果 - 增强版"))
        self.intersection_group.setTitle(self.tr("截面交线"))
        self.fractal_group.setTitle(self.tr("分形维数拟合曲线"))
        self.stats_group.setTitle(self.tr("分形维数统计参数"))
        self.save_btn.setText(self.tr("保存图片"))
        if hasattr(self, 'intersection_lines'):
            self.calculate_and_plot()  



# 完整替换这个类
class SectionResultDialog(QDialog):
    """截面结果显示对话框"""
    
    def __init__(self, intersection_lines, section_type, coordinate_value, cube_size, parent=None):
        super().__init__(parent)
        self.intersection_lines = intersection_lines
        self.section_type = section_type
        self.coordinate_value = coordinate_value
        self.cube_size = cube_size
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        self.setGeometry(100, 100, 1200, 1000)
        layout = QVBoxLayout()
        self.results_group = QGroupBox()
        results_layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton()
        self.save_btn.clicked.connect(self.save_image)
        save_layout.addWidget(self.save_btn)
        save_layout.addStretch()
        results_layout.addLayout(save_layout)
        self.results_group.setLayout(results_layout)
        layout.addWidget(self.results_group)
        self.setLayout(layout)
        self.retranslate_ui()
        self.plot_intersection_lines()
    
    def plot_intersection_lines(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self.section_type == 0: x_coord, y_coord = 1, 2
        elif self.section_type == 1: x_coord, y_coord = 0, 2
        else: x_coord, y_coord = 0, 1
        cube_boundary = np.array([[0, 0], [self.cube_size, 0], [self.cube_size, self.cube_size], [0, self.cube_size], [0, 0]])
        ax.plot(cube_boundary[:, 0], cube_boundary[:, 1], 'k--', linewidth=1, alpha=0.5)
        if len(self.intersection_lines) > 0:
            for line_data in self.intersection_lines:
                points = line_data['points']
                if len(points) >= 2:
                    x_coords = points[:, x_coord]; y_coords = points[:, y_coord]
                    ax.plot(x_coords, y_coords, color='blue', linewidth=2, alpha=0.8)
        ax.set_aspect('equal')
        ax.set_xlim(0, self.cube_size); ax.set_ylim(0, self.cube_size)
        self.figure.tight_layout(); self.canvas.draw()
    
    def update_data(self, intersection_lines, section_type, coordinate_value, cube_size):
        self.intersection_lines = intersection_lines
        self.section_type = section_type
        self.coordinate_value = coordinate_value
        self.cube_size = cube_size
        self.retranslate_ui() # Call retranslate to update titles
        self.plot_intersection_lines()
    
    def save_image(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(self, self.tr("保存截面图片"), "", self.tr("PNG图片 (*.png);;JPG图片 (*.jpg);;PDF文档 (*.pdf)"))
            if filename:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, self.tr("成功"), self.tr("图片已保存到：{0}").format(filename))
        except Exception as e:
            QMessageBox.warning(self, self.tr("错误"), self.tr("保存图片时发生错误：{0}").format(str(e)))
            
    def retranslate_ui(self):
        self.setWindowTitle(self.tr("截面交线结果"))
        self.results_group.setTitle(self.tr("截面交线"))
        self.save_btn.setText(self.tr("保存图片"))

class SliceTrendDialog(QDialog):
    """切片趋势分析对话框 (位置 vs D, 位置 vs N0)"""
    def __init__(self, x_data, d_data, n0_data, direction_label, parent=None):
        super().__init__(parent)
        self.x_data = x_data
        self.d_data = d_data
        self.n0_data = n0_data
        self.direction_label = direction_label
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("切片位置分形特征趋势分析"))
        self.resize(1000, 800)
        
        layout = QVBoxLayout()
        
        # 创建画布
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 保存按钮
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton(self.tr("保存图表"))
        self.save_btn.clicked.connect(self.save_plot)
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        self.plot_data()

    def plot_data(self):
        self.figure.clear()
        
        # 子图1: 位置 vs 分形维数 D
        ax1 = self.figure.add_subplot(211)
        ax1.plot(self.x_data, self.d_data, 'o-', color='#2E86AB', linewidth=2, markersize=6)
        ax1.set_title(self.tr("切片位置 vs 分形维数 (D)"), fontfamily='Microsoft YaHei', fontsize=12, pad=10)
        ax1.set_ylabel(self.tr("分形维数 D"), fontfamily='Microsoft YaHei')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 子图2: 位置 vs 分形初值 N0
        ax2 = self.figure.add_subplot(212)
        ax2.plot(self.x_data, self.n0_data, 's-', color='#E74C3C', linewidth=2, markersize=6)
        ax2.set_title(self.tr("切片位置 vs 分形初值 (N0)"), fontfamily='Microsoft YaHei', fontsize=12, pad=10)
        ax2.set_xlabel(f"{self.direction_label} " + self.tr("坐标位置"), fontfamily='Microsoft YaHei')
        ax2.set_ylabel(self.tr("分形初值 N0"), fontfamily='Microsoft YaHei')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        # 修改对话框提示，明确会保存两个文件
        filename, _ = QFileDialog.getSaveFileName(
            self, self.tr("保存图表及数据"), 
            f"slice_trend_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
            "PNG图片 (*.png);;PDF文档 (*.pdf)"
        )
        if filename:
            try:
                # 1. 保存图片
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                
                # 2. 保存数据 (改为 txt 格式)
                base_name = os.path.splitext(filename)[0]
                txt_filename = f"{base_name}_data.txt"
                
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    # 写入头部信息
                    f.write(f"# {self.windowTitle()}\n")
                    f.write(f"# Generated: {datetime.now().isoformat()}\n")
                    f.write("-" * 40 + "\n")
                    
                    # 写入列名
                    # 判断 x_data 是绝对坐标还是相对位置
                    x_label = "Relative_Position(0-1)" if all(0 <= x <= 1 for x in self.x_data) else "Coordinate"
                    f.write(f"{x_label:<20}\tFractal_Dim(D)\tFractal_N0\n")
                    
                    # 写入数据行
                    for x, d, n0 in zip(self.x_data, self.d_data, self.n0_data):
                        # 处理可能的 None 值
                        d_str = f"{d:.6f}" if d is not None else "N/A"
                        n0_str = f"{n0:.6f}" if n0 is not None else "N/A"
                        f.write(f"{x:<20.6f}\t{d_str}\t{n0_str}\n")
                        
                QMessageBox.information(self, self.tr("成功"), 
                                      self.tr("已成功保存以下文件:\n1. 图片: {0}\n2. 数据: {1}").format(os.path.basename(filename), os.path.basename(txt_filename)))
            except Exception as e:
                QMessageBox.warning(self, self.tr("错误"), str(e))

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = FractalFractureGeneratorGUI()

    # --- 新增代码 ---
    # 创建并关联翻译管理器
    tr_manager = TranslationManager(app)
    window.tr_manager = tr_manager
    # 将语言改变信号连接到UI更新槽函数
    tr_manager.language_changed.connect(window.retranslate_ui)
    # --- 结束新增 ---

    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    main()

# --- END OF REFACTORED FILE ---