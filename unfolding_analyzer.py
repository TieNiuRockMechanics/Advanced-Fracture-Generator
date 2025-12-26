# 文件: unfolding_analyzer.py (最终修正版)

import sys, numpy as np, math, csv
from datetime import datetime
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QTableWidget, QTableWidgetItem, QPushButton, QScrollArea, QHeaderView, QFileDialog, QMessageBox, QLabel)
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QColor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy import stats

# 明确导入为曲线设计的【新版算法】计算器
from fractal_analysis import FractalDimension2DCalculatorForCurves
import matplotlib.pyplot as plt  # <<< 新增：导入此库以设置坐标轴刻度
INSET_FACTOR = 0  # <<< 唯一的修改点


class UnfoldingComputer:
    """
    【调试版修改】UnfoldingComputer 类
    - _get_cuboid_traces 方法被完全重写，以动态计算内缩截面的交线。
    - 内缩因子 inset_factor 在构造函数中被硬编码，便于调试。
    """
    def __init__(self, fractures, cube_size):
        self.fractures = fractures
        self.cube_size = cube_size
        

        #
        self.inset_factor = INSET_FACTOR # <<< 使用顶部的全局设置
        #
        # ##############################################################
        
        self.r = cube_size / 2.0
        self.h_cyl = np.pi * self.r / 2.0

    def get_all_traces(self):
        cuboid_traces = self._get_cuboid_traces()
        cylinder_traces = self._get_cylinder_traces() # 圆柱体逻辑保持不变
        return {**cuboid_traces, **cylinder_traces}

    def _get_cuboid_traces(self):
        """
        【已重写】动态计算立方体内缩截面上的裂缝轨迹。
        此方法不再依赖预计算的 boundary_lines，而是直接调用裂缝对象的
        交线计算方法，以获取由 self.inset_factor 控制的任意位置截面的轨迹。
        """
        traces = {'cuboid_front': [], 'cuboid_right': [], 'cuboid_back': [], 'cuboid_left': []}
        
        # 1. 根据内缩因子 a (self.inset_factor) 计算新的截面位置
        val_near = self.inset_factor * self.cube_size
        val_far = (1.0 - self.inset_factor) * self.cube_size

        # 2. 定义需要计算的四个内缩截面的配置信息
        plane_configs = [
            {'key': 'cuboid_front', 'coord_idx': 1, 'plane_val': val_near},
            {'key': 'cuboid_right', 'coord_idx': 0, 'plane_val': val_far},
            {'key': 'cuboid_back',  'coord_idx': 1, 'plane_val': val_far},
            {'key': 'cuboid_left',  'coord_idx': 0, 'plane_val': val_near},
        ]
        
        tolerance = 1e-6

        # 3. 遍历所有裂缝，为每个裂缝动态计算与这四个截面的交线
        for fracture in self.fractures:
            if not hasattr(fracture, 'clipped_vertices') or fracture.clipped_vertices is None or len(fracture.clipped_vertices) < 3:
                continue
            
            vertices = fracture.clipped_vertices
            
            for config in plane_configs:
                key = config['key']
                
                # 动态调用交线计算函数
                intersection_points_list = fracture._calculate_polygon_plane_intersection(
                    vertices, config['coord_idx'], config['plane_val'], tolerance
                )

                if len(intersection_points_list) >= 2:
                    points_3d = np.array(intersection_points_list)
                    points_2d = []
                    
                    # 4. 应用与旧代码完全相同的2D投影和展开变换逻辑
                    if key == 'cuboid_front':
                        points_2d = [(p[0], p[2]) for p in points_3d]
                    elif key == 'cuboid_right':
                        points_2d = [(p[1], p[2]) for p in points_3d]
                    elif key == 'cuboid_back':
                        points_2d = [(-p[0] + self.cube_size, p[2]) for p in points_3d]
                    elif key == 'cuboid_left':
                        points_2d = [(self.cube_size - p[1], p[2]) for p in points_3d]
                    
                    if points_2d:
                        points_2d.sort(key=lambda p: p[0])
                        traces[key].append(points_2d)

        return traces

    def _get_cylinder_traces(self):
        traces = {'cylinder_front': [], 'cylinder_right': [], 'cylinder_back': [], 'cylinder_left': []}
        angle_map = {
            'cylinder_right': (-np.pi / 4, np.pi / 4), 'cylinder_back': (np.pi / 4, 3 * np.pi / 4),
            'cylinder_left': (3 * np.pi / 4, 5 * np.pi / 4), 'cylinder_front': (5 * np.pi / 4, 7 * np.pi / 4)
        }
        for fracture in self.fractures:
            intersections = self._calculate_ellipse_cylinder_intersection(fracture)
            for key, (angle_min, angle_max) in angle_map.items():
                for trace_3d in intersections:
                    points_2d = []
                    for t, z in trace_3d:
                        in_range = (key == 'cylinder_right' and (t >= (angle_min + 2*np.pi) or t < angle_max)) or \
                                   (angle_min <= t < angle_max)
                        if in_range:
                            center_angle = (angle_min + angle_max) / 2.0; current_angle = t
                            if key == 'cylinder_right' and current_angle > np.pi: current_angle -= 2*np.pi
                            x_centered = (current_angle - center_angle) * self.r; points_2d.append((x_centered, z))
                    if len(points_2d) >= 2: points_2d.sort(key=lambda p: p[0]); traces[key].append(points_2d)
        return traces

    def _calculate_ellipse_cylinder_intersection(self, fracture):
        intersection_curves = []
        n_points = 1000  # <--- 在这里设置了采样点数量
        C = fracture.center
        N = fracture.normal
        if abs(N[2]) < 1e-9: return []
        rot_matrix = np.linalg.inv(fracture._transform_ellipse_points(np.eye(3)))
        center_x, center_y = self.cube_size / 2.0, self.cube_size / 2.0
        t_values = np.linspace(0, 2 * np.pi, n_points)
        x_cyl = self.r * np.cos(t_values) + center_x; y_cyl = self.r * np.sin(t_values) + center_y
        z_intersect = C[2] - (N[0]*(x_cyl - C[0]) + N[1]*(y_cyl - C[1])) / N[2]
        points_on_plane = np.vstack([x_cyl, y_cyl, z_intersect]).T
        valid_points_3d = []; is_in_segment = False
        for p in points_on_plane:
            if not (center_y - self.h_cyl/2 <= p[2] <= center_y + self.h_cyl/2):
                if is_in_segment: intersection_curves.append(valid_points_3d); valid_points_3d = []; is_in_segment = False
                continue
            p_local = rot_matrix @ (p - C)
            if (p_local[0]/fracture.semi_major_axis)**2 + (p_local[1]/fracture.semi_minor_axis)**2 <= 1.0:
                if not is_in_segment: is_in_segment = True
                angle = np.arctan2(p[1] - center_y, p[0] - center_x)
                if angle < 0: angle += 2 * np.pi
                valid_points_3d.append((angle, p[2] - (center_y - self.h_cyl/2)))
            elif is_in_segment:
                intersection_curves.append(valid_points_3d); valid_points_3d = []; is_in_segment = False
        if valid_points_3d: intersection_curves.append(valid_points_3d)
        return intersection_curves

def _calculate_average_from_results_robust(results_list):
    """【修正】使用更稳健的 np.polyfit 进行拟合，并返回用于拟合的平均点"""
    levels_data = {}
    for result in results_list:
        # 安全地获取 levels 列表，如果不存在则视为空列表
        for level in result.get('levels', []):
            bs = level['box_size']
            levels_data.setdefault(bs, []).append(level['valid_count'])
    
    # 如果没有任何数据，返回空结果
    if not levels_data:
        return {'D': None, 'N0': None, 'R2': None, 'avg_points': []}

    # 计算每个 box_size 的平均计数值
    avg_points = [{'box_size': bs, 'valid_count': np.mean(cs)} for bs, cs in sorted(levels_data.items(), reverse=True)]
    
    # 筛选出有效计数值大于0的点用于拟合
    valid_data = [(p['box_size'], p['valid_count']) for p in avg_points if p['valid_count'] > 0]
    
    # 如果有效数据点少于2个，无法进行线性拟合
    if len(valid_data) < 2:
        return {'D': None, 'N0': None, 'R2': None, 'avg_points': avg_points}

    log_sizes = np.log10([d[0] for d in valid_data])
    log_counts = np.log10([d[1] for d in valid_data])
    
    try:
        # 检查是否存在无穷大或NaN值
        if np.isinf(log_counts).any() or np.isnan(log_counts).any():
            return {'D': None, 'N0': None, 'R2': None, 'avg_points': avg_points}
            
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        y_pred = np.polyval(coeffs, log_sizes)
        y_mean = np.mean(log_counts)
        ss_res = np.sum((log_counts - y_pred) ** 2)
        ss_tot = np.sum((log_counts - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # ★ 核心修改：在返回结果中增加 'avg_points' 键
        return {'D': -slope, 'N0': 10**intercept, 'R2': r_squared, 'avg_points': avg_points}
        
    except np.linalg.LinAlgError:
        return {'D': None, 'N0': None, 'R2': None, 'avg_points': avg_points}

# 文件: unfolding_analyzer.py

# 用这个【修正版】完整替换旧的 perform_unfolding_analysis 函数
def perform_unfolding_analysis(fractures, cube_size, gen_iterations=4):
    """
    【最终修正版】执行表面展开分析的核心函数。
    - 只对立方体轨迹应用裁剪逻辑。
    - 对圆柱体轨迹只进行平移，不裁剪，以确保与UI分析逻辑一致。
    """
    inset_factor = INSET_FACTOR
    gen_iterations = max(1, gen_iterations - 1)

    def _clip_line_to_box_2d(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
        # ... (裁剪函数代码保持不变) ...
        INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
        def compute_code(x, y):
            code = INSIDE
            if x < x_min: code |= LEFT
            elif x > x_max: code |= RIGHT
            if y < y_min: code |= BOTTOM
            elif y > y_max: code |= TOP
            return code
        code1, code2 = compute_code(x1, y1), compute_code(x2, y2)
        while True:
            if not (code1 | code2): return (x1, y1), (x2, y2)
            if code1 & code2: return None
            code_out = code1 if code1 else code2; x, y = 0, 0
            if code_out & TOP:
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1) if y1 != y2 else x1; y = y_max
            elif code_out & BOTTOM:
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1) if y1 != y2 else x1; y = y_min
            elif code_out & RIGHT:
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1) if x1 != x2 else y1; x = x_max
            elif code_out & LEFT:
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1) if x1 != x2 else y1; x = x_min
            if code_out == code1: x1, y1, code1 = x, y, compute_code(x, y)
            else: x2, y2, code2 = x, y, compute_code(x, y)

    computer = UnfoldingComputer(fractures, cube_size)
    all_traces = computer.get_all_traces()
    fractal_results = {}
    side_length = computer.h_cyl

    for name, traces in all_traces.items():
        if not traces:
            fractal_results[name] = {'D': None, 'N0': None, 'R2': None, 'levels': []}
            continue

        dim, fit_data, levels = None, None, []
        
        # ★★★ 核心修正：在这里也进行逻辑分离 ★★★
        if name.startswith('cuboid'):
            original_width = computer.cube_size
            crop_min = inset_factor * original_width
            crop_max = (1.0 - inset_factor) * original_width
            new_side_length = crop_max - crop_min

            if new_side_length > 1e-9:
                cropped_and_translated_traces = []
                for trace in traces:
                    if len(trace) < 2: continue
                    for i in range(len(trace) - 1):
                        p1, p2 = trace[i], trace[i+1]
                        clipped_segment = _clip_line_to_box_2d(p1[0], p1[1], p2[0], p2[1], crop_min, crop_min, crop_max, crop_max)
                        if clipped_segment:
                            cp1, cp2 = clipped_segment
                            translated_p1 = (cp1[0] - crop_min, cp1[1] - crop_min)
                            translated_p2 = (cp2[0] - crop_min, cp2[1] - crop_min)
                            cropped_and_translated_traces.append([translated_p1, translated_p2])
                
                if cropped_and_translated_traces:
                    calculator = FractalDimension2DCalculatorForCurves(iterations=gen_iterations)
                    dim, fit_data, levels = calculator.calculate_fractal_dimension(
                        cropped_and_translated_traces, new_side_length, new_side_length
                    )
        
        else: # name.startswith('cylinder')
            width = height = side_length
            translated_traces = []
            x_min_of_square = -width / 2.0
            y_min_of_square = 0.0
            for trace in traces:
                translated_traces.append([(p[0] - x_min_of_square, p[1] - y_min_of_square) for p in trace])
            
            calculator = FractalDimension2DCalculatorForCurves(iterations=gen_iterations)
            dim, fit_data, levels = calculator.calculate_fractal_dimension(translated_traces, width, height)

        # 统一封装结果
        if dim is not None:
            fractal_results[name] = {'D': dim, 'N0': fit_data['fractal_n0'], 'R2': fit_data['r_squared'], 'levels': levels}
        else:
            fractal_results[name] = {'D': None, 'N0': None, 'R2': None, 'levels': []}

    cuboid_avg = _calculate_average_from_results_robust([v for k, v in fractal_results.items() if k.startswith('cuboid')])
    cylinder_avg = _calculate_average_from_results_robust([v for k, v in fractal_results.items() if k.startswith('cylinder')])
    
    return {
        'cuboid_unfolding_dim': cuboid_avg.get('D'), 'cuboid_unfolding_n0': cuboid_avg.get('N0'),
        'cylinder_unfolding_dim': cylinder_avg.get('D'), 'cylinder_unfolding_n0': cylinder_avg.get('N0'),
    }

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi); self.axes = self.fig.add_subplot(111); super().__init__(self.fig)

class UnfoldingAnalysisWidget(QWidget):
    # ==================== 替换 __init__ 函数 ====================
    def __init__(self, main_window, parent=None):
        """【已修正】构造函数，接收并保存主窗口的引用。"""
        super().__init__(parent)
        self.main_window = main_window  # 保存主窗口的引用
        self.computer = None
        self.fractal_results = {}
        self.canvases = {}
        self.init_ui()
    
    def tr(self, text): return QCoreApplication.translate("UnfoldingAnalysisWidget", text)

    def init_ui(self):
        main_layout = QVBoxLayout(self); scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True); main_layout.addWidget(scroll_area)
        container_widget = QWidget(); scroll_area.setWidget(container_widget); container_layout = QVBoxLayout(container_widget)
        plots_group = QGroupBox(self.tr("二维展开轨迹图")); plots_layout = QGridLayout(plots_group); container_layout.addWidget(plots_group)
        self.face_info = {
            'cuboid_front': {'title': self.tr('立方体-前面'), 'row': 0, 'col': 0}, 'cuboid_right': {'title': self.tr('立方体-右面'), 'row': 0, 'col': 1},
            'cuboid_back':  {'title': self.tr('立方体-后面'), 'row': 0, 'col': 2}, 'cuboid_left':  {'title': self.tr('立方体-左面'), 'row': 0, 'col': 3},
            'cylinder_front': {'title': self.tr('圆柱体-前面'), 'row': 1, 'col': 0}, 'cylinder_right': {'title': self.tr('圆柱体-右面'), 'row': 1, 'col': 1},
            'cylinder_back':  {'title': self.tr('圆柱体-后面'), 'row': 1, 'col': 2}, 'cylinder_left':  {'title': self.tr('圆柱体-左面'), 'row': 1, 'col': 3},
        }
        for name, info in self.face_info.items(): self.canvases[name] = MplCanvas(self); plots_layout.addWidget(self.canvases[name], info['row'], info['col'])
        results_group = QGroupBox(self.tr("分形维数计算结果")); results_layout = QVBoxLayout(results_group); container_layout.addWidget(results_group)
        self.results_table = QTableWidget(); self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([self.tr("展开面"), self.tr("分形维数 (D)"), self.tr("分形初值 (N0)"), self.tr("拟合优度 (R²)")])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); results_layout.addWidget(self.results_table)
        self.export_btn = QPushButton(self.tr("导出数据")); self.export_btn.clicked.connect(self._export_results); container_layout.addWidget(self.export_btn)
        self.clear_results()

    def update_analysis(self, fractures, cube_size):
        self.computer = UnfoldingComputer(fractures, cube_size); self.run_analysis(); self.export_btn.setEnabled(True)

    def clear_results(self):
        self.computer = None; self.fractal_results = {}
        for name, canvas in self.canvases.items(): canvas.axes.clear(); canvas.axes.set_title(self.face_info[name]['title']); canvas.axes.text(0.5, 0.5, self.tr("等待数据..."), ha='center', va='center', transform=canvas.axes.transAxes); canvas.draw()
        self.results_table.setRowCount(0); self.export_btn.setEnabled(False)

    def run_analysis(self):
        if not self.computer: self.clear_results(); return
        all_traces = self.computer.get_all_traces(); self._plot_all_traces(all_traces)
        self.fractal_results = self._calculate_all_fractal_dimensions_ui(all_traces)
        self.fractal_results['cuboid_average'] = self._calculate_average_fractal_dimension_ui([k for k in self.face_info if k.startswith('cuboid')])
        self.fractal_results['cylinder_average'] = self._calculate_average_fractal_dimension_ui([k for k in self.face_info if k.startswith('cylinder')])
        self._update_results_table()

# 文件: unfolding_analyzer.py (在 UnfoldingAnalysisWidget 类中)

    # 用这个新版本完整替换旧的 _plot_all_traces 函数
# 文件: unfolding_analyzer.py (在 UnfoldingAnalysisWidget 类中)

    # 用这个【最终隔离修复版】完整替换 _plot_all_traces 函数
    def _plot_all_traces(self, all_traces):
        side_length = self.computer.h_cyl
        
        # 将裁剪算法定义在函数内部，仅供立方体逻辑使用
        def _clip_line_to_box_2d(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
            INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
            def compute_code(x, y):
                code = INSIDE
                if x < x_min: code |= LEFT
                elif x > x_max: code |= RIGHT
                if y < y_min: code |= BOTTOM
                elif y > y_max: code |= TOP
                return code
            code1, code2 = compute_code(x1, y1), compute_code(x2, y2)
            while True:
                if not (code1 | code2): return (x1, y1), (x2, y2)
                if code1 & code2: return None
                code_out = code1 if code1 else code2; x, y = 0, 0
                if code_out & TOP:
                    x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1) if y1 != y2 else x1; y = y_max
                elif code_out & BOTTOM:
                    x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1) if y1 != y2 else x1; y = y_min
                elif code_out & RIGHT:
                    y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1) if x1 != x2 else y1; x = x_max
                elif code_out & LEFT:
                    y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1) if x1 != x2 else y1; x = x_min
                if code_out == code1: x1, y1, code1 = x, y, compute_code(x, y)
                else: x2, y2, code2 = x, y, compute_code(x, y)

        inset_factor = INSET_FACTOR

        for name, traces in all_traces.items():
            ax = self.canvases[name].axes
            ax.clear()

            # ★★★ 核心修改：根据名称选择不同的绘图逻辑 ★★★
            if name.startswith('cuboid'):
                # --- 这是新的、只针对立方体的裁剪绘图逻辑 ---
                original_width = self.computer.cube_size
                crop_min = inset_factor * original_width
                crop_max = (1.0 - inset_factor) * original_width
                new_side_length = crop_max - crop_min
                
                if new_side_length <= 1e-9: # 使用容差判断
                    ax.set_title(self.face_info[name]['title'])
                    ax.text(0.5, 0.5, self.tr("无效裁剪区域 (a>=0.5)"), ha='center', va='center', transform=ax.transAxes)
                    self.canvases[name].draw()
                    continue

                ax.set_xlim(0, new_side_length)
                ax.set_ylim(0, new_side_length)

                if traces:
                    for trace in traces:
                        if len(trace) < 2: continue
                        for i in range(len(trace) - 1):
                            p1 = trace[i]; p2 = trace[i+1]
                            clipped_segment = _clip_line_to_box_2d(
                                p1[0], p1[1], p2[0], p2[1],
                                crop_min, crop_min, crop_max, crop_max
                            )
                            if clipped_segment:
                                cp1, cp2 = clipped_segment
                                translated_p1 = (cp1[0] - crop_min, cp1[1] - crop_min)
                                translated_p2 = (cp2[0] - crop_min, cp2[1] - crop_min)
                                ax.plot([translated_p1[0], translated_p2[0]], 
                                        [translated_p1[1], translated_p2[1]], 'b-')

            else: # name.startswith('cylinder')
                # --- 这是旧的、完全不变的圆柱体绘图逻辑 ---
                width = side_length
                ax.set_xlim(0, width)
                ax.set_ylim(0, width)
                # ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
                # ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

                for trace in traces:
                    if not trace: continue
                    x_raw, y_coords = zip(*trace)
                    # 将圆柱图的x坐标平移，使其从0开始，便于绘图
                    x_coords = [x + width / 2.0 for x in x_raw]
                    ax.plot(x_coords, y_coords, 'b-')

            ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

            ax.set_title(self.face_info[name]['title'])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linestyle='--', alpha=0.6)
            self.canvases[name].draw()

# 文件: unfolding_analyzer.py (在 UnfoldingAnalysisWidget 类中)

    # 用这个【最终隔离修复版】完整替换 _calculate_all_fractal_dimensions_ui 函数
    def _calculate_all_fractal_dimensions_ui(self, all_traces):
        """【已修正】只对立方体应用裁剪逻辑进行分形计算，圆柱体保持原样。"""
        fractal_results = {}
        side_length = self.computer.h_cyl
        
        gen_iterations = 2
        if self.main_window and hasattr(self.main_window, 'iterations_spin'):
            gen_iterations = self.main_window.iterations_spin.value()
        gen_iterations = max(1, gen_iterations - 1)

        
        def _clip_line_to_box_2d(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
            # ... (裁剪函数代码保持不变) ...
            INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
            def compute_code(x, y):
                code = INSIDE
                if x < x_min: code |= LEFT
                elif x > x_max: code |= RIGHT
                if y < y_min: code |= BOTTOM
                elif y > y_max: code |= TOP
                return code
            code1, code2 = compute_code(x1, y1), compute_code(x2, y2)
            while True:
                if not (code1 | code2): return (x1, y1), (x2, y2)
                if code1 & code2: return None
                code_out = code1 if code1 else code2; x, y = 0, 0
                if code_out & TOP:
                    x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1) if y1 != y2 else x1; y = y_max
                elif code_out & BOTTOM:
                    x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1) if y1 != y2 else x1; y = y_min
                elif code_out & RIGHT:
                    y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1) if x1 != x2 else y1; x = x_max
                elif code_out & LEFT:
                    y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1) if x1 != x2 else y1; x = x_min
                if code_out == code1: x1, y1, code1 = x, y, compute_code(x, y)
                else: x2, y2, code2 = x, y, compute_code(x, y)

        inset_factor = INSET_FACTOR
            
        for name, traces in all_traces.items():
            if not traces:
                fractal_results[name] = {'D': None, 'N0': None, 'R2': None, 'levels': []}
                continue

            dim, fit_data, levels = None, None, []

            # ★★★ 核心修改：根据名称选择不同的计算逻辑 ★★★
            if name.startswith('cuboid'):
                # --- 这是新的、只针对立方体的裁剪计算逻辑 ---
                original_width = self.computer.cube_size
                crop_min = inset_factor * original_width
                crop_max = (1.0 - inset_factor) * original_width
                new_side_length = crop_max - crop_min

                if new_side_length > 1e-9:
                    cropped_and_translated_traces = []
                    for trace in traces:
                        if len(trace) < 2: continue
                        for i in range(len(trace) - 1):
                            p1, p2 = trace[i], trace[i+1]
                            clipped_segment = _clip_line_to_box_2d(p1[0], p1[1], p2[0], p2[1], crop_min, crop_min, crop_max, crop_max)
                            if clipped_segment:
                                cp1, cp2 = clipped_segment
                                translated_p1 = (cp1[0] - crop_min, cp1[1] - crop_min)
                                translated_p2 = (cp2[0] - crop_min, cp2[1] - crop_min)
                                cropped_and_translated_traces.append([translated_p1, translated_p2])
                    
                    if cropped_and_translated_traces:
                        calculator = FractalDimension2DCalculatorForCurves(iterations=gen_iterations)
                        dim, fit_data, levels = calculator.calculate_fractal_dimension(
                            cropped_and_translated_traces, new_side_length, new_side_length
                        )
            
            else: # name.startswith('cylinder')
                # ★★★ START OF MODIFICATION: 按照您的思路修改圆柱体逻辑 ★★★
                width = height = side_length
                translated_traces = []

                # 圆柱展开后的正方形，其坐标范围是 x in [-width/2, width/2], y in [0, height]
                # 因此，其左下角坐标是 (-width/2, 0)
                x_min_of_square = -width / 2.0
                y_min_of_square = 0.0

                for trace in traces:
                    # 使用固定的正方形左下角坐标进行平移
                    translated_traces.append([(p[0] - x_min_of_square, p[1] - y_min_of_square) for p in trace])
                # ★★★ END OF MODIFICATION ★★★
                
                calculator = FractalDimension2DCalculatorForCurves(iterations=gen_iterations)
                # 传入的 width 和 height 依然是完整的正方形尺寸，这保证了结果的正确性
                dim, fit_data, levels = calculator.calculate_fractal_dimension(translated_traces, width, height)

            # --- 统一的结果封装 ---
            if dim is not None:
                fractal_results[name] = {'D': dim, 'N0': fit_data['fractal_n0'], 'R2': fit_data['r_squared'], 'levels': levels}
            else:
                fractal_results[name] = {'D': None, 'N0': None, 'R2': None, 'levels': []}
        
        return fractal_results

    def _calculate_average_fractal_dimension_ui(self, face_keys):
        return _calculate_average_from_results_robust([self.fractal_results[key] for key in face_keys if key in self.fractal_results])

    def _update_results_table(self):
        display_order = ['cuboid_front', 'cuboid_right', 'cuboid_back', 'cuboid_left', 'cuboid_average', 'cylinder_front', 'cylinder_right', 'cylinder_back', 'cylinder_left', 'cylinder_average']
        self.results_table.setRowCount(len(display_order))
        for i, name in enumerate(display_order):
            result = self.fractal_results.get(name, {})
            title = (self.tr('立方体-平均值') if 'cuboid' in name else self.tr('圆柱体-平均值')) if name.endswith('_average') else self.face_info[name]['title']
            self.results_table.setItem(i, 0, QTableWidgetItem(title))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{result.get('D'):.4f}" if result.get('D') is not None else self.tr("N/A")))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{result.get('N0'):.4f}" if result.get('N0') is not None else self.tr("N/A")))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{result.get('R2'):.4f}" if result.get('R2') is not None else self.tr("N/A")))
            if name.endswith('_average'):
                for col in range(self.results_table.columnCount()): self.results_table.item(i, col).setBackground(QColor(230, 245, 255))


# 文件: unfolding_analyzer.py (在 UnfoldingAnalysisWidget 类中)

    def _export_results(self):
        """
        将汇总数据导出为CSV，所有绘图导出为TIF图片，
        并将所有分形计算的原始盒计数数据合并到一个TXT文件中。
        """
        directory = QFileDialog.getExistingDirectory(self, self.tr("选择导出目录"))
        if not directory:
            return

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"unfolding_analysis_{timestamp}"

            # 3. 导出CSV表格数据 (此部分不变)
            csv_path = os.path.join(directory, f"{base_filename}_data.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                headers = [self.results_table.horizontalHeaderItem(i).text() for i in range(self.results_table.columnCount())]
                writer.writerow(headers)
                for row in range(self.results_table.rowCount()):
                    writer.writerow([self.results_table.item(row, col).text() for col in range(self.results_table.columnCount())])

            # ★★★★★ 新增功能：导出详细的levels数据到TXT文件 ★★★★★
            txt_path = os.path.join(directory, f"{base_filename}_fractal_levels.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"--- {self.tr('分形维数计算详细数据')} ---\n")
                f.write(f"{self.tr('文件生成时间')}: {timestamp}\n")
                f.write("="*40 + "\n\n")

                # 使用与表格相同的显示顺序
                display_order = ['cuboid_front', 'cuboid_right', 'cuboid_back', 'cuboid_left', 'cuboid_average', 
                                 'cylinder_front', 'cylinder_right', 'cylinder_back', 'cylinder_left', 'cylinder_average']

                for name in display_order:
                    result = self.fractal_results.get(name, {})
                    
                    # 获取标题
                    if name.endswith('_average'):
                        title = self.tr('立方体-平均值') if 'cuboid' in name else self.tr('圆柱体-平均值')
                    else:
                        title = self.face_info[name]['title']
                    
                    f.write(f"--- {title} ---\n")
                    
                    # 写入该面的分形参数结果
                    d_val = f"{result.get('D'):.4f}" if result.get('D') is not None else "N/A"
                    n0_val = f"{result.get('N0'):.4f}" if result.get('N0') is not None else "N/A"
                    r2_val = f"{result.get('R2'):.4f}" if result.get('R2') is not None else "N/A"
                    f.write(f"{self.tr('分形维数 (D)')}: {d_val}\n")
                    f.write(f"{self.tr('分形初值 (N0)')}: {n0_val}\n")
                    f.write(f"{self.tr('拟合优度 (R²)')}: {r2_val}\n\n")
                    
                    f.write(f"{'Box Size (L)':<20} | {'Count (N)':<20}\n")
                    f.write(f"{'-'*20} | {'-'*20}\n")
                    
                    # 根据是平均值还是单个面，从不同的键获取数据
                    if name.endswith('_average'):
                        # 对于平均值，我们从 'avg_points' 获取数据
                        level_data = result.get('avg_points', [])
                    else:
                        # 对于单个面，我们从 'levels' 获取数据
                        level_data = result.get('levels', [])
                    
                    if not level_data:
                        f.write(self.tr("无有效数据\n"))
                    else:
                        for level in level_data:
                            f.write(f"{level['box_size']:<20.6f} | {level['valid_count']:<20.4f}\n")
                    
                    f.write("\n" + "="*40 + "\n\n")

            # 4. 循环导出所有8张画布上的图片 (此部分不变)
            for name, canvas in self.canvases.items():
                image_path = os.path.join(directory, f"{base_filename}_plot_{name}.tif")
                canvas.fig.savefig(image_path, dpi=500, format='tif', bbox_inches='tight')

            # 5. 显示成功信息 (★ 修改提示信息以包含TXT文件)
            success_message = self.tr("结果已成功导出到以下目录:\n{0}\n\n"
                                      "导出的文件包括:\n"
                                      "- 1个汇总数据的CSV文件\n"
                                      "- 1个详细计算过程的TXT文件\n"
                                      "- 8张轨迹图的TIF图片").format(directory)
            QMessageBox.information(self, self.tr("成功"), success_message)

        except Exception as e:
            error_message = self.tr("导出文件时发生错误:\n{0}").format(str(e))
            QMessageBox.critical(self, self.tr("错误"), error_message)