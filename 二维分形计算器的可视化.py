# 文件名: 二维分形计算器的可视化_最终版.py
# (已修正核心计数逻辑错误, 并根据用户最新需求增强可视化功能)

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import traceback
import datetime
import random

# ==============================================================================
# 步骤0: 字体与国际化设置
# ==============================================================================

# ----- 默认使用英文 -----
USE_ENGLISH_LABELS = True

# --- 新增: 设置英文字体为 Times New Roman ---
if USE_ENGLISH_LABELS:
    try:
        plt.rcParams['font.family'] = 'serif' # 告诉 Matplotlib 优先使用衬线字体
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        # 为了让公式和普通文本看起来协调，推荐设置数学字体
        plt.rcParams['mathtext.fontset'] = 'stix' 
        print("字体 'Times New Roman' 设置成功。")
    except Exception as e:
        print(f"警告: 设置 'Times New Roman' 失败，将使用默认字体。错误: {e}")

# --- 中文字体设置 (作为备用) ---
try:
    # 即使默认英文，也尝试加载中文，以便在需要时切换
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    print("中文字体 'SimHei' 加载成功。")
except Exception:
    if not USE_ENGLISH_LABELS:
        print("警告: 未找到中文字体 'SimHei'，将自动切换到英文标签。")
        USE_ENGLISH_LABELS = True

def tr(chinese_text, english_text):
    """一个简单的翻译函数，根据全局标志返回相应的语言。"""
    return english_text if USE_ENGLISH_LABELS else chinese_text

# ==============================================================================
# 步骤1: 整合所有必需的核心类 (保持不变)
# ==============================================================================

BOX_SIZE_PRECISION = 6
LENGTH_PRECISION = 6

class FractalDimension2DCalculator:
    """二维分形维数计算器的基类。"""
    def __init__(self, iterations=6, initial_box_size=None, generation_iterations=3):
        self.iterations = iterations
        self.initial_box_size = initial_box_size
        self.generation_iterations = generation_iterations
        self.levels = []
        self.all_segments = []

    def _create_box_grid(self, level, box_size):
        """创建盒子网格"""
        boxes = []
        grid_count = 2 ** level
        for i in range(grid_count):
            for j in range(grid_count):
                min_x, max_x = i * box_size, (i + 1) * box_size
                min_y, max_y = j * box_size, (j + 1) * box_size
                boxes.append({'id': i * grid_count + j, 'min_x': round(min_x, BOX_SIZE_PRECISION), 'max_x': round(max_x, BOX_SIZE_PRECISION), 'min_y': round(min_y, BOX_SIZE_PRECISION), 'max_y': round(max_y, BOX_SIZE_PRECISION)})
        return boxes

    def _clip_line_to_box(self, x1, y1, x2, y2, min_x, min_y, max_x, max_y):
        """Cohen-Sutherland线段裁切算法"""
        def _compute_outcode(x, y):
            code = 0
            if x < min_x: code |= 1
            elif x > max_x: code |= 2
            if y < min_y: code |= 4
            elif y > max_y: code |= 8
            return code
        
        outcode1, outcode2 = _compute_outcode(x1, y1), _compute_outcode(x2, y2)
        
        while True:
            if not (outcode1 | outcode2):
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                return [{'point1': (x1, y1), 'point2': (x2, y2), 'length': length}] if length > 1e-9 else []
            if outcode1 & outcode2:
                return []
            
            outcode_out = outcode1 or outcode2
            x, y = 0.0, 0.0

            if outcode_out & 8:
                x = x1 + (x2 - x1) * (max_y - y1) / (y2 - y1) if y1 != y2 else x1; y = max_y
            elif outcode_out & 4:
                x = x1 + (x2 - x1) * (min_y - y1) / (y2 - y1) if y1 != y2 else x1; y = min_y
            elif outcode_out & 2:
                y = y1 + (y2 - y1) * (max_x - x1) / (x2 - x1) if x1 != x2 else y1; x = max_x
            elif outcode_out & 1:
                y = y1 + (y2 - y1) * (min_x - x1) / (x2 - x1) if x1 != x2 else y1; x = min_x
            
            if outcode_out == outcode1: x1, y1, outcode1 = x, y, _compute_outcode(x, y)
            else: x2, y2, outcode2 = x, y, _compute_outcode(x, y)
    
    def _fit_fractal_dimension(self):
        """从levels数据中拟合分形维数。"""
        if len(self.levels) < 2: return None, None, self.levels
        valid_data = [(l['box_size'], l['valid_count']) for l in self.levels if l['valid_count'] > 0]
        if len(valid_data) < 2: return None, None, self.levels
        
        valid_box_sizes, valid_counts = zip(*valid_data)
        log_sizes, log_counts = np.log10(valid_box_sizes), np.log10(valid_counts)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dimension = -coeffs[0]
        y_pred = np.polyval(coeffs, log_sizes)
        ss_res, ss_tot = np.sum((log_counts - y_pred) ** 2), np.sum((log_counts - np.mean(log_counts)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        fit_data = {'log_sizes': log_sizes, 'log_counts': log_counts, 'coeffs': coeffs, 'r_squared': r_squared}
        return fractal_dimension, fit_data, self.levels

class AdvancedFractalSimulator(FractalDimension2DCalculator):
    """从 FD-Advanced.py 移植并修正的核心分形维数模拟器"""
    def __init__(self, iterations, D, N0, L0, angle_base, angle_range, use_second_group=False, angle_base2=0, angle_range2=0, group1_ratio=0.5, use_random_angles=True):
        super().__init__(iterations=iterations, initial_box_size=L0, generation_iterations=iterations)
        self.D, self.N0, self.L0 = D, N0, L0
        self.angle_base, self.angle_range = angle_base, angle_range
        self.use_second_group, self.angle_base2, self.angle_range2 = use_second_group, angle_base2, angle_range2
        self.group1_ratio, self.use_random_angles = group1_ratio, use_random_angles
        self._progress_callback = print

    def calculate_theoretical_cracks(self, box_size):
        return int(round(self.N0 * (box_size ** (-self.D))))

    def run_simulation(self, progress_callback=None):
        if progress_callback: self._progress_callback = progress_callback
        self.levels, self.all_segments = [], []
        self._progress_callback(f"Fractal Params: D={self.D}, N0={self.N0}, L0={self.L0}")

        L_current, L_previous = self.L0, 2 * self.L0
        N_theoretical = self.calculate_theoretical_cracks(L_current)
        boxes = self._create_box_grid(0, L_current)
        existing_count, g1, g2 = self.count_valid_segments_angle_distribution(boxes, L_current)
        new_needed = max(0, N_theoretical - existing_count)
        group_alloc = self._calculate_group_allocation(N_theoretical, new_needed, g1, g2)
        initial_segments = self.generate_constrained_segments_in_area(0, 0, self.L0, new_needed, L_current, L_previous, boxes, group_alloc) if new_needed > 0 else []
        
        self.levels.append({'level': 0, 'box_size': L_current, 'theoretical_count': N_theoretical, 'actual_count': existing_count + len(initial_segments), 'existing_count': existing_count, 'new_count': len(initial_segments), 'boxes': boxes, 'new_segments': initial_segments})
        self.all_segments.extend(initial_segments)
        self._progress_callback(f"Level 0 finished: Generated {len(initial_segments)} new fractures.")

        for iteration in range(1, self.iterations + 1):
            L_current, L_previous = self.L0 / (2**iteration), self.L0 / (2**(iteration-1))
            N_theoretical = self.calculate_theoretical_cracks(L_current)
            boxes = self._create_box_grid(iteration, L_current)
            existing_count, g1, g2 = self.count_valid_segments_angle_distribution(boxes, L_current)
            new_needed = max(0, N_theoretical - existing_count)
            group_alloc = self._calculate_group_allocation(N_theoretical, new_needed, g1, g2)
            new_segments = self.generate_constrained_segments_in_area(0, 0, self.L0, new_needed, L_current, L_previous, boxes, group_alloc) if new_needed > 0 else []
            
            level_info = {'level': iteration, 'box_size': L_current, 'theoretical_count': N_theoretical, 'existing_count': existing_count, 'new_count': len(new_segments), 'actual_count': existing_count + len(new_segments), 'boxes': boxes, 'new_segments': new_segments}
            self.levels.append(level_info)
            self.all_segments.extend(new_segments)
            self._progress_callback(f"Level {iteration} finished: Added {len(new_segments)} new fractures, Total valid: {level_info['actual_count']}.")
        return self.levels
    
    def _calculate_group_allocation(self, N_theoretical, new_segments_needed, existing_group1, existing_group2):
        if self.use_second_group and new_segments_needed > 0 and not self.use_random_angles:
            target_g1, target_g2 = int(N_theoretical * self.group1_ratio), N_theoretical - int(N_theoretical * self.group1_ratio)
            needed_g1, needed_g2 = max(0, target_g1 - existing_group1), max(0, target_g2 - existing_group2)
            total_needed = needed_g1 + needed_g2
            if total_needed > new_segments_needed:
                scale = new_segments_needed / total_needed if total_needed > 0 else 0
                needed_g1, needed_g2 = int(needed_g1 * scale), new_segments_needed - int(needed_g1 * scale)
            elif total_needed < new_segments_needed:
                diff = new_segments_needed - total_needed
                if existing_group1 + needed_g1 <= existing_group2 + needed_g2: needed_g1 += diff
                else: needed_g2 += diff
            return (needed_g1, needed_g2)
        return None

    def generate_constrained_segments_in_area(self, start_x, start_y, area_size, count, min_length, max_length, boxes, group_allocation):
        actual_min, actual_max = min_length, max_length if max_length is not None else math.sqrt(2) * area_size
        if actual_min >= actual_max: return []
        if self.use_random_angles:
            return self._generate_single_group_segments(start_x, start_y, area_size, count, actual_min, actual_max, 0, 0, boxes, min_length, "Random Angle")
        elif self.use_second_group:
            g1_count, g2_count = group_allocation if group_allocation is not None else (int(count * self.group1_ratio), count - int(count * self.group1_ratio))
            g1_segs = self._generate_single_group_segments(start_x, start_y, area_size, g1_count, actual_min, actual_max, self.angle_base, self.angle_range, boxes, min_length, "Group 1")
            g2_segs = self._generate_single_group_segments(start_x, start_y, area_size, g2_count, actual_min, actual_max, self.angle_base2, self.angle_range2, boxes, min_length, "Group 2")
            return g1_segs + g2_segs
        else:
            return self._generate_single_group_segments(start_x, start_y, area_size, count, actual_min, actual_max, self.angle_base, self.angle_range, boxes, min_length, "Single Group")

    def _generate_single_group_segments(self, start_x, start_y, area_size, count, actual_min, actual_max, angle_base, angle_range, boxes, min_length, group_name):
        segments, attempts = [], 0
        if count <= 0: return []
        max_attempts = 20000 * count
        angle_min, angle_max = (0, 2*math.pi) if self.use_random_angles else (math.radians(angle_base - angle_range), math.radians(angle_base + angle_range))
        while len(segments) < count and attempts < max_attempts:
            attempts += 1
            length, angle = random.uniform(actual_min * 1.01, actual_max), random.uniform(angle_min, angle_max)
            x1, y1 = random.uniform(start_x, start_x + area_size), random.uniform(start_y, start_y + area_size)
            x2, y2 = x1 + length * math.cos(angle), y1 + length * math.sin(angle)
            if any(c['length'] > min_length for box in boxes for c in self._clip_line_to_box(x1, y1, x2, y2, box['min_x'], box['min_y'], box['max_x'], box['max_y'])):
                segments.append({'point1': (round(x1, LENGTH_PRECISION), round(y1, LENGTH_PRECISION)), 'point2': (round(x2, LENGTH_PRECISION), round(y2, LENGTH_PRECISION)), 'length': round(length, LENGTH_PRECISION), 'angle': round(math.degrees(angle), 2), 'level': len(self.levels), 'angle_group': 1 if angle_base == self.angle_base and not self.use_random_angles else 2, 'is_random_angle': self.use_random_angles})
        return segments

    def count_valid_segments_angle_distribution(self, boxes, min_length):
        total_count, group1_count, group2_count = 0, 0, 0
        for segment in self.all_segments:
            x1, y1 = segment['point1']
            x2, y2 = segment['point2']
            for box in boxes:
                clipped_segments = self._clip_line_to_box(x1, y1, x2, y2, box['min_x'], box['min_y'], box['max_x'], box['max_y'])
                for clipped in clipped_segments:
                    if clipped['length'] > min_length:
                        total_count += 1
                        if not self.use_random_angles:
                            angle_group = segment.get('angle_group', 1)
                            if angle_group == 1:
                                group1_count += 1
                            else:
                                group2_count += 1
        return total_count, group1_count, group2_count

class VisualLengthBasedCalculator(FractalDimension2DCalculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.highlight_info = {}

    def calculate_fractal_dimension(self, line_segments, cube_size, face_name=None):
        self.levels, self.all_segments = [], [{'id': i, 'points': seg} for i, seg in enumerate(line_segments)]
        L0 = cube_size if self.initial_box_size is None else self.initial_box_size
        for iteration in range(self.generation_iterations + 1):
            L_current = round(L0 / (2 ** iteration), 6)
            boxes = self._create_box_grid(iteration, L_current)
            valid_count = self._count_valid_segments_in_boxes(self.all_segments, boxes, L_current)
            self.levels.append({'level': iteration, 'box_size': L_current, 'valid_count': valid_count})
        return self._fit_fractal_dimension()

    def _count_valid_segments_in_boxes(self, segments, boxes, min_length):
        total_count = 0
        current_level = int(round(math.log2(self.initial_box_size / min_length)))
        self.highlight_info[current_level] = {}
        for box in boxes:
            box_id, count_for_this_box, segments_info = box['id'], 0, []
            for segment in segments:
                p1, p2 = segment['points']
                clipped = self._clip_line_to_box(p1[0], p1[1], p2[0], p2[1], box['min_x'], box['min_y'], box['max_x'], box['max_y'])
                if not clipped: continue
                is_valid = sum(c['length'] for c in clipped) >= min_length
                if is_valid: count_for_this_box += 1
                for c in clipped: segments_info.append({'start': c['point1'], 'end': c['point2'], 'length': c['length'], 'is_valid': is_valid})
            if segments_info: self.highlight_info[current_level][box_id] = {'count': count_for_this_box, 'segments': segments_info}
            total_count += count_for_this_box
        return total_count

class VisualTraditionalBoxCountingCalculator(FractalDimension2DCalculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.highlight_info = {}

    def calculate_fractal_dimension(self, line_segments, cube_size, face_name=None):
        self.levels, self.all_segments = [], [{'id': i, 'points': seg} for i, seg in enumerate(line_segments)]
        L0 = cube_size if self.initial_box_size is None else self.initial_box_size
        for iteration in range(self.generation_iterations + 1):
            L_current = round(L0 / (2 ** iteration), 6)
            boxes = self._create_box_grid(iteration, L_current)
            valid_count = self._count_intersected_boxes(self.all_segments, boxes, iteration)
            self.levels.append({'level': iteration, 'box_size': L_current, 'valid_count': valid_count})
        return self._fit_fractal_dimension()

    def _count_intersected_boxes(self, segments, boxes, level):
        self.highlight_info[level] = {'hit_boxes': set(), 'hit_fractures': set()}
        for box in boxes:
            box_hit = False
            for segment in segments:
                p1, p2 = segment['points']
                if self._clip_line_to_box(p1[0], p1[1], p2[0], p2[1], box['min_x'], box['min_y'], box['max_x'], box['max_y']):
                    box_hit = True
                    self.highlight_info[level]['hit_fractures'].add(segment['id'])
            if box_hit: self.highlight_info[level]['hit_boxes'].add(box['id'])
        return len(self.highlight_info[level]['hit_boxes'])

# ==============================================================================
# 步骤4: 辅助函数 (已修改)
# ==============================================================================

def refit_for_plotting(levels_data, levels_to_plot):
    """
    根据用户选择的可视化级别，重新计算拟合数据用于绘图。
    """
    if not levels_data or not levels_to_plot:
        return None
    
    # 筛选出用户选择的级别的数据
    filtered_levels = [levels_data[i] for i in levels_to_plot if i < len(levels_data) and levels_data[i]['valid_count'] > 0]
    
    if len(filtered_levels) < 2: # 数据点少于2个无法拟合
        return None

    valid_box_sizes = [l['box_size'] for l in filtered_levels]
    valid_counts = [l['valid_count'] for l in filtered_levels]

    log_sizes, log_counts = np.log10(valid_box_sizes), np.log10(valid_counts)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    
    y_pred = np.polyval(coeffs, log_sizes)
    ss_res = np.sum((log_counts - y_pred) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 返回一个新的 fit_data 字典
    return {'log_sizes': log_sizes, 'log_counts': log_counts, 'coeffs': coeffs, 'r_squared': r_squared}


def print_results(title, dim, fit_data, levels):
    print(f"\n{'='*50}\n {title}\n{'='*50}")
    if dim is not None:
        n0 = 10**fit_data.get('coeffs', [0, 0])[1]
        r_squared = fit_data.get('r_squared', 0.0)
        print(f"  {tr('分形维数 (D)', 'Fractal Dimension (D)')} = {dim:.6f}, N0 = {n0:.6f}, R² = {r_squared:.6f}")
        header_level, header_box_size, header_count = tr('层级', 'Level'), tr('盒子尺寸 (L)', 'Box Size (L)'), tr('有效计数 (N)', 'Count (N)')
        print(f"  {header_level:<10} | {header_box_size:<20} | {header_count:<15}\n  {'-'*10} | {'-'*20} | {'-'*15}")
        for i, level in enumerate(levels):
            print(f"  {i:<10} | {level['box_size']:<20.6f} | {level['valid_count']:<15}")
    else: print(f"  {tr('计算失败或数据不足。', 'Calculation failed or data is insufficient.')}")
    print("="*50)

# ----- 修改: 彻底重构拟合图函数 -----
def plot_fit_graph(ax, title, fit_data, show_n0=True):
    """在指定的ax上绘制log-log拟合散点图, 具有最终布局。"""
    ax.clear()
    if not fit_data:
        ax.set_title(title, fontsize=12)
        ax.text(0.5, 0.5, tr("无有效数据", "No valid data"), ha='center', va='center')
        return

    log_sizes = fit_data['log_sizes']
    log_counts = fit_data['log_counts']
    coeffs = fit_data['coeffs']
    r_squared = fit_data['r_squared']
    
    # 绘制散点和拟合线
    ax.scatter(log_sizes, log_counts, label=tr('数据点', 'Data Points'), zorder=5)
    y_pred = np.polyval(coeffs, log_sizes)
    ax.plot(log_sizes, y_pred, 'r-', label=tr('线性拟合', 'Linear Fit'), zorder=4)
    
    # --- 准备标题和图例信息 ---
    D = -coeffs[0]
    intercept = coeffs[1]
    
    # 1. 构建结果字符串，用于标题
    if show_n0:
        N0 = 10**intercept
        results_str = f"$D = {D:.4f}$, $N_0 = {N0:.2f}$, $R^2 = {r_squared:.4f}$"
    else:
        results_str = f"$D = {D:.4f}$, $R^2 = {r_squared:.4f}$"
    
    # 2. 将结果放入标题
    ax.set_title(f"{title}\n{results_str}", fontsize=10)

    # 3. 构建公式字符串，作为图例的标题
    formula_str = f"$\\log_{{10}}(N) = {-D:.4f} \\cdot \\log_{{10}}(L) + {intercept:.4f}$"

    # 4. 创建一个合并的图例，位于右上角
    ax.legend(title=formula_str, loc='upper right', fontsize='small', frameon=False)

    ax.set_xlabel("$\\log_{10}(\\varepsilon)$")
    ax.set_ylabel("$\\log_{10}(N)$")
    ax.grid(True, linestyle='--', alpha=0.6)

# ----- 修改: 更新可视化标题格式 -----
def plot_visualization(ax, title, network, level_data, highlight_info, level_to_plot, method_type):
    box_size = level_data[level_to_plot]['box_size']
    cube_size = level_data[0]['box_size']
    count_at_level = level_data[level_to_plot]['valid_count']
    
    ax.clear()
    for fracture in network: ax.plot([fracture[0][0], fracture[1][0]], [fracture[0][1], fracture[1][1]], color='lightgray', linewidth=0.8, zorder=1)
    grid_count = int(math.ceil(cube_size / box_size))
    if method_type == 'length_based':
        for i, j in np.ndindex(grid_count, grid_count):
            box_min_x, box_min_y = i * box_size, j * box_size
            rect = patches.Rectangle((box_min_x, box_min_y), box_size, box_size, linewidth=0.5, edgecolor='black', facecolor='none', zorder=2); ax.add_patch(rect)
            box_info = highlight_info.get(level_to_plot, {}).get(i * grid_count + j)
            if box_info:
                count = box_info.get('count', 0)
                if count > 0:
                    rect.set_facecolor('lightgreen'); rect.set_alpha(0.4)
                    ax.text(box_min_x + box_size * 0.05, box_min_y + box_size * 0.95, str(count), color='red', fontsize=12, weight='bold', ha='left', va='top', zorder=6)

                for seg in box_info.get('segments', []):
                                    p1, p2 = seg['start'], seg['end']
                                    color = 'blue' if seg['is_valid'] else 'orange'
                                    lw = 2.5 if seg['is_valid'] else 1.0
                                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=lw, zorder=3)
                                    
                                    # --- 新增：计算并修正文本位置 ---
                                    text_x = (p1[0] + p2[0]) / 2
                                    text_y = (p1[1] + p2[1]) / 2
                                    
                                    # 定义一个小的边界百分比
                                    padding = 0.03 * cube_size
                                    
                                    # 使用 np.clip 将坐标限制在 (padding, cube_size - padding) 范围内
                                    clipped_x = np.clip(text_x, padding, cube_size - padding)
                                    clipped_y = np.clip(text_y, padding, cube_size - padding)

                                    ax.text(clipped_x, clipped_y, f"{seg['length']:.2f}", fontsize=7, color='black', 
                                            ha='center', va='center', 
                                            bbox=dict(boxstyle='round,pad=0.1', fc='yellow', ec='none', alpha=0.7), zorder=5)

    elif method_type == 'traditional':
        level_info = highlight_info.get(level_to_plot, {}); hit_fractures = level_info.get('hit_fractures', set())
        for idx, fracture in enumerate(network):
            if idx in hit_fractures: ax.plot([fracture[0][0], fracture[1][0]], [fracture[0][1], fracture[1][1]], color='blue', linewidth=2.5, zorder=3)
        hit_boxes = level_info.get('hit_boxes', set())
        for i, j in np.ndindex(grid_count, grid_count):
            box_min_x, box_min_y = i * box_size, j * box_size
            rect = patches.Rectangle((box_min_x, box_min_y), box_size, box_size, linewidth=0.5, edgecolor='black', facecolor='none', zorder=4); ax.add_patch(rect)
            if i * grid_count + j in hit_boxes:
                rect.set_facecolor('lightblue'); rect.set_alpha(0.5)
                ax.text(box_min_x + box_size * 0.05, box_min_y + box_size * 0.95, '1', color='red', fontsize=12, weight='bold', ha='left', va='top', zorder=6)
    
    ax.set_aspect('equal', adjustable='box'); ax.set_xlim(0, cube_size); ax.set_ylim(0, cube_size)
    
    # --- 修改后的标题格式逻辑 ---
    if USE_ENGLISH_LABELS:
        iter_str = f"Iteration = {level_to_plot}" # <--- 这是新行
        epsilon_str = f"$\\epsilon_{{({level_to_plot})}} = {box_size:.4f}$"
        count_str = f"$N_{{(\\epsilon)}} = {count_at_level}$"
        full_title = f"{title}\n{iter_str}, {epsilon_str}, {count_str}"
    else:
        title_level, title_box_size, title_count = '级别', '尺寸', '计数'
        full_title = f"{title}\n{title_level} {level_to_plot}, {title_box_size} = {box_size:.4f}, {title_count} = {count_at_level}"
    
    ax.set_title(full_title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


# ==============================================================================
# 步骤5: 创建Tkinter UI界面 (已修改)
# ==============================================================================
class VisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title(tr("二维分形裂缝网络可视化工具", "2D Fractal Fracture Network Visualization Tool"))
        self.fig, self.axes = None, None
        self.fracture_network = None
        self.calculator1, self.levels1, self.fit_results1 = None, None, None
        self.calculator2, self.levels2, self.fit_results2 = None, None, None
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        fractal_frame = ttk.LabelFrame(main_frame, text=tr("分形生成参数", "Fractal Generation Parameters"), padding="10"); fractal_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Label(fractal_frame, text=tr("分形维数 (D):", "Dimension (D):")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.d_var = tk.DoubleVar(value=1.2); ttk.Entry(fractal_frame, textvariable=self.d_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(fractal_frame, text=tr("参考数量 (N0):", "Initial Count (N0):")).grid(row=0, column=2, sticky=tk.W, padx=5)
        self.n0_var = tk.IntVar(value=2); ttk.Entry(fractal_frame, textvariable=self.n0_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        ttk.Label(fractal_frame, text=tr("参考尺寸 (L0):", "Ref. Size (L0):")).grid(row=1, column=0, sticky=tk.W, padx=5)
        self.l0_var = tk.DoubleVar(value=1.0); ttk.Entry(fractal_frame, textvariable=self.l0_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(fractal_frame, text=tr("生成迭代次数:", "Gen. Iterations:")).grid(row=1, column=2, sticky=tk.W, padx=5)
        self.iter_var = tk.IntVar(value=4); ttk.Entry(fractal_frame, textvariable=self.iter_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5)
        angle_frame = ttk.LabelFrame(main_frame, text=tr("角度参数", "Angle Parameters"), padding="10"); angle_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.angle_mode = tk.StringVar(value="random")
        ttk.Radiobutton(angle_frame, text=tr("完全随机", "Random"), variable=self.angle_mode, value="random", command=self._update_angle_controls_state).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(angle_frame, text=tr("单组角度", "Single Group"), variable=self.angle_mode, value="single", command=self._update_angle_controls_state).grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(angle_frame, text=tr("双组角度", "Dual Group"), variable=self.angle_mode, value="dual", command=self._update_angle_controls_state).grid(row=2, column=0, sticky=tk.W)
        self.g1_frame = ttk.Frame(angle_frame); self.g1_frame.grid(row=1, column=1, sticky=tk.W)
        ttk.Label(self.g1_frame, text=tr("组1: 基础", "G1: Base")).pack(side=tk.LEFT, padx=2)
        self.g1_base_var = tk.DoubleVar(value=30); ttk.Entry(self.g1_frame, textvariable=self.g1_base_var, width=6).pack(side=tk.LEFT)
        ttk.Label(self.g1_frame, text=tr("± 范围", "± Range")).pack(side=tk.LEFT, padx=2)
        self.g1_range_var = tk.DoubleVar(value=10); ttk.Entry(self.g1_frame, textvariable=self.g1_range_var, width=6).pack(side=tk.LEFT)
        self.g2_frame = ttk.Frame(angle_frame); self.g2_frame.grid(row=2, column=1, sticky=tk.W)
        ttk.Label(self.g2_frame, text=tr("组2: 基础", "G2: Base")).pack(side=tk.LEFT, padx=2)
        self.g2_base_var = tk.DoubleVar(value=120); ttk.Entry(self.g2_frame, textvariable=self.g2_base_var, width=6).pack(side=tk.LEFT)
        ttk.Label(self.g2_frame, text=tr("± 范围", "± Range")).pack(side=tk.LEFT, padx=2)
        self.g2_range_var = tk.DoubleVar(value=10); ttk.Entry(self.g2_frame, textvariable=self.g2_range_var, width=6).pack(side=tk.LEFT)
        ttk.Label(self.g2_frame, text=tr("组1占比(%)", "G1 Ratio(%)")).pack(side=tk.LEFT, padx=2)
        self.g1_ratio_var = tk.IntVar(value=50); ttk.Entry(self.g2_frame, textvariable=self.g1_ratio_var, width=6).pack(side=tk.LEFT)
        vis_frame = ttk.LabelFrame(main_frame, text=tr("可视化参数", "Visualization Parameters"), padding="10"); vis_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(vis_frame, text=tr("可视化级别:", "Levels to Plot:")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.levels_entry = ttk.Entry(vis_frame, width=30); self.levels_entry.insert(0, "0 1 2"); self.levels_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.run_button = ttk.Button(main_frame, text=tr("运行并可视化", "Run & Visualize"), command=self._run_visualization); self.run_button.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10, padx=5)
        self.export_button = ttk.Button(main_frame, text=tr("导出全部图片", "Export All Images"), command=self._export_images, state=tk.DISABLED); self.export_button.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=10, padx=5)
        self.root.columnconfigure(0, weight=1); main_frame.columnconfigure(1, weight=1)
        self._update_angle_controls_state()
        
    def _update_angle_controls_state(self):
        mode = self.angle_mode.get()
        state_g1 = tk.NORMAL if mode in ['single', 'dual'] else tk.DISABLED
        state_g2 = tk.NORMAL if mode == 'dual' else tk.DISABLED
        for child in self.g1_frame.winfo_children(): child.configure(state=state_g1)
        for child in self.g2_frame.winfo_children(): child.configure(state=state_g2)
        
    def _run_visualization(self): 
        try:
            levels_to_plot = [int(x) for x in self.levels_entry.get().strip().split()]
            if not levels_to_plot: messagebox.showerror(tr("输入错误", "Input Error"), tr("请输入至少一个可视化级别。", "Please enter at least one level to plot.")); return
            ITERATIONS = max(self.iter_var.get(), max(levels_to_plot))
            angle_mode = self.angle_mode.get()
            params = {'iterations': ITERATIONS, 'D': self.d_var.get(), 'N0': self.n0_var.get(), 'L0': self.l0_var.get(), 'angle_base': self.g1_base_var.get(), 'angle_range': self.g1_range_var.get(), 'use_second_group': angle_mode == "dual", 'angle_base2': self.g2_base_var.get(), 'angle_range2': self.g2_range_var.get(), 'group1_ratio': self.g1_ratio_var.get() / 100.0, 'use_random_angles': angle_mode == "random"}
            simulator = AdvancedFractalSimulator(**params); simulator.run_simulation()
            self.fracture_network = [(seg['point1'], seg['point2']) for seg in simulator.all_segments]
            
            self.calculator1 = VisualLengthBasedCalculator(generation_iterations=ITERATIONS, initial_box_size=params['L0'])
            dim1, fit_data1, self.levels1 = self.calculator1.calculate_fractal_dimension(self.fracture_network, params['L0'])
            self.fit_results1 = (dim1, fit_data1)
            
            self.calculator2 = VisualTraditionalBoxCountingCalculator(generation_iterations=ITERATIONS, initial_box_size=params['L0'])
            dim2, fit_data2, self.levels2 = self.calculator2.calculate_fractal_dimension(self.fracture_network, params['L0'])
            self.fit_results2 = (dim2, fit_data2)
            
            print_results(tr("方法1: 基于长度的盒计数法", "Method 1: Length-Based Box Counting"), dim1, fit_data1, self.levels1)
            print_results(tr("方法2: 传统盒维数法", "Method 2: Traditional Box Counting"), dim2, fit_data2, self.levels2)
            
            if plt.get_fignums(): plt.close('all')
            
            num_cols = len(levels_to_plot)
            self.fig, self.axes = plt.subplots(2, num_cols + 1, figsize=(6 * (num_cols + 1), 12), dpi=150)
            if num_cols == 0: self.axes = self.axes.reshape(2, 1) 
            
            self.fig.suptitle(tr("两种二维分形维数计算方法的可视化对比", "Visualization Comparison of Two 2D Fractal Dimension Methods"), fontsize=16)
            
            for i, level in enumerate(levels_to_plot):
                ax1 = self.axes[0, i] if (num_cols + 1) > 1 else self.axes[0]
                plot_visualization(ax1, tr("方法1: 基于长度", "Method 1: Length-Based"), self.fracture_network, self.levels1, self.calculator1.highlight_info, level, 'length_based')
                
                ax2 = self.axes[1, i] if (num_cols + 1) > 1 else self.axes[1]
                plot_visualization(ax2, tr("方法2: 传统方法", "Method 2: Traditional"), self.fracture_network, self.levels2, self.calculator2.highlight_info, level, 'traditional')

            plot_fit_data1 = refit_for_plotting(self.levels1, levels_to_plot)
            plot_fit_data2 = refit_for_plotting(self.levels2, levels_to_plot)

            fit_ax1 = self.axes[0, -1]
            plot_fit_graph(fit_ax1, tr("方法1: 拟合结果", "Method 1: Fit Result"), plot_fit_data1, show_n0=True)
            fit_ax1.set_box_aspect(1) 

            fit_ax2 = self.axes[1, -1]
            plot_fit_graph(fit_ax2, tr("方法2: 拟合结果", "Method 2: Fit Result"), plot_fit_data2, show_n0=False)
            fit_ax2.set_box_aspect(1)

            self.fig.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局
            self.export_button.config(state=tk.NORMAL); plt.show()
        except Exception: messagebox.showerror(tr("运行时错误", "Runtime Error"), f"{tr('发生未知错误:', 'An unexpected error occurred:')}\n{traceback.format_exc()}")
        
    def _export_images(self):
        if not self.fig: messagebox.showwarning(tr("警告", "Warning"), tr("请先运行可视化。", "Please run the visualization first.")); return
        directory = filedialog.askdirectory(title=tr("选择导出文件夹", "Select Export Directory"))
        if not directory: return
        try:
            levels_to_plot = [int(x) for x in self.levels_entry.get().strip().split()]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for i, level in enumerate(levels_to_plot):
                temp_fig1, temp_ax1 = plt.subplots(figsize=(4, 4), dpi=500); plot_visualization(temp_ax1, tr("方法1: 基于长度", "Method 1: Length-Based"), self.fracture_network, self.levels1, self.calculator1.highlight_info, level, 'length_based'); temp_fig1.savefig(os.path.join(directory, f"{timestamp}_level_{level}_method1_length_based.png"), bbox_inches='tight'); plt.close(temp_fig1)
                temp_fig2, temp_ax2 = plt.subplots(figsize=(4, 4), dpi=500); plot_visualization(temp_ax2, tr("方法2: 传统方法", "Method 2: Traditional"), self.fracture_network, self.levels2, self.calculator2.highlight_info, level, 'traditional'); temp_fig2.savefig(os.path.join(directory, f"{timestamp}_level_{level}_method2_traditional.png"), bbox_inches='tight'); plt.close(temp_fig2)
            
            # --- 修改: 确保导出的拟合图也是方形的 ---
            if self.fit_results1 and self.fit_results1[1]:
                fit_fig1, fit_ax1 = plt.subplots(figsize=(4, 4), dpi=500)
                plot_fit_graph(fit_ax1, tr("方法1: 拟合结果", "Method 1: Fit Result"), self.fit_results1[1], show_n0=True)
                fit_ax1.set_box_aspect(1)
                fit_fig1.savefig(os.path.join(directory, f"{timestamp}_method1_fit_result.png"), bbox_inches='tight')
                plt.close(fit_fig1)

            if self.fit_results2 and self.fit_results2[1]:
                fit_fig2, fit_ax2 = plt.subplots(figsize=(4, 4), dpi=500)
                plot_fit_graph(fit_ax2, tr("方法2: 拟合结果", "Method 2: Fit Result"), self.fit_results2[1], show_n0=False)
                fit_ax2.set_box_aspect(1)
                fit_fig2.savefig(os.path.join(directory, f"{timestamp}_method2_fit_result.png"), bbox_inches='tight')
                plt.close(fit_fig2)
            
            messagebox.showinfo(tr("成功", "Success"), f"{tr('所有图片已成功导出到:', 'All images have been successfully exported to:')}\n{directory}")
        except Exception: messagebox.showerror(tr("导出失败", "Export Failed"), f"{tr('导出过程中发生错误:', 'An error occurred during export:')}\n{traceback.format_exc()}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualizationApp(root)
    root.mainloop()