# 文件: fractal_analysis.py (最终修复版 - 已修正裁剪算法)

import numpy as np
import math
import open3d as o3d
from scipy import stats

NUM_SLICES_PER_DIRECTION = 101


class FractalDimension3DCalculator:
    """三维分形维数计算器（基于盒计数法）"""
    
    def __init__(self, iterations=6):
        self.iterations = iterations
        self.levels = []
        self.progress_callback = None
    
    def calculate_fractal_dimension_3d(self, fractures, cube_size):
        self.levels = []
        if not fractures: return None, None, []
        
        for iteration in range(self.iterations + 1):
            box_size = round(cube_size / (2 ** iteration), 6)
            if self.progress_callback: self.progress_callback(f"正在计算第 {iteration + 1}/{self.iterations + 1} 层级 (盒子大小: {box_size:.6f})...")
            boxes = self._generate_boxes(iteration, box_size, cube_size)
            valid_count = self._count_valid_boxes(fractures, boxes, box_size)
            level_info = {'level': iteration, 'box_size': box_size, 'valid_count': valid_count}
            self.levels.append(level_info)
            if self.progress_callback: self.progress_callback(f"第 {iteration + 1} 层级完成: 有效计数 {valid_count}/{len(boxes)}")
        
        return self._fit_fractal_dimension_3d()
    
    def _generate_boxes(self, level, box_size, cube_size):
        boxes = []
        grid_count = 2 ** level
        for i in range(grid_count):
            for j in range(grid_count):
                for k in range(grid_count):
                    min_x = round(i * box_size, 6); max_x = round(min((i + 1) * box_size, cube_size), 6)
                    min_y = round(j * box_size, 6); max_y = round(min((j + 1) * box_size, cube_size), 6)
                    min_z = round(k * box_size, 6); max_z = round(min((k + 1) * box_size, cube_size), 6)
                    box = {'min': np.array([min_x, min_y, min_z]), 'max': np.array([max_x, max_y, max_z])}
                    boxes.append(box)
        return boxes
    
    def _count_valid_boxes(self, fractures, boxes, box_size):
        threshold_area = box_size ** 2; total_counts = 0
        fracture_bounds = [(np.min(f.vertices, axis=0), np.max(f.vertices, axis=0)) if len(f.vertices) > 0 else (None, None) for f in fractures]
        for box in boxes:
            box_min, box_max = box['min'], box['max']; box_counts = 0
            for i, fracture in enumerate(fractures):
                frac_min, frac_max = fracture_bounds[i]
                if frac_min is None: continue
                if (frac_max[0] < box_min[0] or frac_min[0] > box_max[0] or
                    frac_max[1] < box_min[1] or frac_min[1] > box_max[1] or
                    frac_max[2] < box_min[2] or frac_min[2] > box_max[2]):
                    continue
                if fracture.get_area_in_box(box_min, box_max) >= threshold_area:
                    box_counts += 1
            total_counts += box_counts
        return total_counts
    
    def _fit_fractal_dimension_3d(self):
        if len(self.levels) < 2: return None, None, self.levels
        valid_data = [(l['box_size'], l['valid_count']) for l in self.levels if l['valid_count'] > 0]
        if len(valid_data) < 2: return None, None, self.levels
        valid_box_sizes, valid_counts = zip(*valid_data)
        log_sizes = np.log10(valid_box_sizes); log_counts = np.log10(valid_counts)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dimension = -coeffs[0]
        fit_data = {'log_sizes': log_sizes, 'log_counts': log_counts, 'coeffs': coeffs, 'r_squared': self._calculate_r_squared(log_sizes, log_counts, coeffs)}
        return fractal_dimension, fit_data, self.levels
    
    def _calculate_r_squared(self, x, y, coeffs):
        y_pred = np.polyval(coeffs, x); y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2); ss_tot = np.sum((y - y_mean) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

class FractalDimension2DCalculator:
    """二维裂缝分形维数计算器（裁剪算法已修复）"""
    def __init__(self, iterations=6, initial_box_size=None, generation_iterations=3):
        self.iterations = iterations
        self.initial_box_size = initial_box_size
        self.generation_iterations = generation_iterations
        self.levels = []
        self.progress_callback = None

    def calculate_fractal_dimension(self, line_segments, cube_size, face_name=None):
        self.levels = []
        L0 = cube_size if self.initial_box_size is None else self.initial_box_size
        
        processed_segments = []
        for line_points in line_segments:
            if len(line_points) >= 2:
                for i in range(len(line_points) - 1):
                    x1, y1 = line_points[i][:2]; x2, y2 = line_points[i + 1][:2]
                    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    processed_segments.append({'point1': (x1, y1), 'point2': (x2, y2), 'length': length})
        
        if not processed_segments: 
            return None, None, []
            
        for iteration in range(self.generation_iterations + 1):
            L_current = round(L0 / (2 ** iteration), 6)

            if self.progress_callback: 
                self.progress_callback(f"正在计算第 {iteration + 1}/{self.generation_iterations + 1} 层级 (盒子大小: {L_current:.6f})...")
            
            boxes = self._create_box_grid(iteration, L_current, cube_size)
            valid_count = self._count_valid_segments_in_boxes(processed_segments, boxes, L_current)
            level_info = {'level': iteration, 'box_size': L_current, 'valid_count': valid_count, 'grid_count': 2 ** iteration}
                
            self.levels.append(level_info)
            
        return self._fit_fractal_dimension()

    def _create_box_grid(self, level, box_size, total_size):
        boxes = []; grid_count = int(math.ceil(total_size / box_size))
        for i in range(grid_count):
            for j in range(grid_count):
                min_x = round(i * box_size, 6); max_x = round(min((i + 1) * box_size, total_size), 6)
                min_y = round(j * box_size, 6); max_y = round(min((j + 1) * box_size, total_size), 6)
                boxes.append({'min': np.array([min_x, min_y]), 'max': np.array([max_x, max_y]), 'id': i * grid_count + j})
        return boxes

    def _count_valid_segments_in_boxes(self, segments, boxes, min_length):
        total_count = 0
        for segment in segments:
            x1, y1 = segment['point1']; x2, y2 = segment['point2']
            for box in boxes:
                min_x, min_y = box['min']; max_x, max_y = box['max']
                clipped_segments = self._clip_line_to_box(x1, y1, x2, y2, min_x, min_y, max_x, max_y)
                total_length_in_box = sum(seg['length'] for seg in clipped_segments)
                if total_length_in_box >= min_length: total_count += 1
        return total_count

    # ##########################################################################
    # ### START OF MODIFICATION ###
    # ##########################################################################
    # 用正确、健壮的迭代式裁剪算法完整替换旧的 _clip_line_to_box 函数
    def _clip_line_to_box(self, x1_orig, y1_orig, x2_orig, y2_orig, min_x, min_y, max_x, max_y):
        """
        【已修复】使用来自方法2的、完整且正确的Cohen-Sutherland迭代算法。
        """
        def compute_code(x, y):
            code = 0
            if x < min_x: code |= 1  # 左
            elif x > max_x: code |= 2  # 右
            if y < min_y: code |= 4  # 下
            elif y > max_y: code |= 8  # 上
            return code

        x1, y1, x2, y2 = x1_orig, y1_orig, x2_orig, y2_orig
        code1, code2 = compute_code(x1, y1), compute_code(x2, y2)

        while True:
            # 情况1: 两个端点都在盒子内部，接受线段
            if not (code1 | code2):
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                return [{'start': (x1, y1), 'end': (x2, y2), 'length': length}]
            
            # 情况2: 两个端点都在盒子的同一外侧，拒绝线段
            if code1 & code2:
                return []

            # 情况3: 需要裁剪
            code_out = code1 if code1 else code2
            x, y = 0, 0

            # 计算交点
            if code_out & 8:  # 点在上方
                x = x1 + (x2 - x1) * (max_y - y1) / (y2 - y1) if y1 != y2 else x1
                y = max_y
            elif code_out & 4:  # 点在下方
                x = x1 + (x2 - x1) * (min_y - y1) / (y2 - y1) if y1 != y2 else x1
                y = min_y
            elif code_out & 2:  # 点在右方
                y = y1 + (y2 - y1) * (max_x - x1) / (x2 - x1) if x1 != x2 else y1
                x = max_x
            elif code_out & 1:  # 点在左方
                y = y1 + (y2 - y1) * (min_x - x1) / (x2 - x1) if x1 != x2 else y1
                x = min_x
            
            # 将外部点移动到交点，然后继续循环
            if code_out == code1:
                x1, y1 = x, y
                code1 = compute_code(x1, y1)
            else:
                x2, y2 = x, y
                code2 = compute_code(x2, y2)
    # ##########################################################################
    # ### END OF MODIFICATION ###
    # ##########################################################################

    def _fit_fractal_dimension(self):
        if len(self.levels) < 2: return None, None, self.levels
        valid_data = [(l['box_size'], l['valid_count']) for l in self.levels if l['valid_count'] > 0]
        if len(valid_data) < 2: return None, None, self.levels
        valid_box_sizes, valid_counts = zip(*valid_data)
        log_sizes = np.log10(valid_box_sizes); log_counts = np.log10(valid_counts)
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dimension = -coeffs[0]
        fit_data = {'log_sizes': log_sizes, 'log_counts': log_counts, 'coeffs': coeffs, 'r_squared': self._calculate_r_squared(log_sizes, log_counts, coeffs)}
        return fractal_dimension, fit_data, self.levels

    def _calculate_r_squared(self, x, y, coeffs):
        y_pred = np.polyval(coeffs, x); y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2); ss_tot = np.sum((y - y_mean) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

class FractalDimension2DCalculatorForCurves:
    """【新版曲线算法】二维裂缝分形维数计算器"""
    def __init__(self, iterations=4, min_box_size=0.001):
        self.iterations = iterations
        self.min_box_size = min_box_size
        self.progress_callback = None

    def _clip_segment(self, p1, p2, box_min, box_max):
        def compute_code(x, y):
            code = 0
            if x < box_min[0]: code |= 1
            elif x > box_max[0]: code |= 2
            if y < box_min[1]: code |= 4
            elif y > box_max[1]: code |= 8
            return code
        x1, y1 = p1; x2, y2 = p2
        code1, code2 = compute_code(x1, y1), compute_code(x2, y2)
        while True:
            if not (code1 | code2): return (x1, y1), (x2, y2)
            if code1 & code2: return None
            code_out = code1 if code1 else code2
            x, y = 0, 0
            if code_out & 8: x = x1 + (x2 - x1) * (box_max[1] - y1) / (y2 - y1) if y1 != y2 else x1; y = box_max[1]
            elif code_out & 4: x = x1 + (x2 - x1) * (box_min[1] - y1) / (y2 - y1) if y1 != y2 else x1; y = box_min[1]
            elif code_out & 2: y = y1 + (y2 - y1) * (box_max[0] - x1) / (x2 - x1) if x1 != x2 else y1; x = box_max[0]
            elif code_out & 1: y = y1 + (y2 - y1) * (box_min[0] - x1) / (x2 - x1) if x1 != x2 else y1; x = box_min[0]
            if code_out == code1: x1, y1 = x, y; code1 = compute_code(x1, y1)
            else: x2, y2 = x, y; code2 = compute_code(x2, y2)

    def _calculate_length_in_box(self, trajectory, box_min, box_max):
        if len(trajectory) < 2: return 0.0
        total_length = 0.0
        for i in range(len(trajectory) - 1):
            clipped = self._clip_segment(trajectory[i], trajectory[i+1], box_min, box_max)
            if clipped:
                cp1, cp2 = clipped
                total_length += math.sqrt((cp2[0] - cp1[0])**2 + (cp2[1] - cp1[1])**2)
        return total_length

    def _count_total_for_iteration(self, trajectories, width, height, box_size):
        total_count = 0
        nx, ny = max(1, math.ceil(width / box_size)), max(1, math.ceil(height / box_size))
        boxes = [{'min': (i * box_size, j * box_size), 'max': ((i + 1) * box_size, (j + 1) * box_size)} for i in range(nx) for j in range(ny)]
        for trajectory in trajectories:
            for box in boxes:
                if self._calculate_length_in_box(trajectory, box['min'], box['max']) >= box_size:
                    total_count += 1
        return total_count

    def calculate_fractal_dimension(self, trajectories, width, height):
        levels = []
        initial_box_size = max(width, height)
        box_sizes = [initial_box_size / (2**i) for i in range(self.iterations + 1)]
        for box_size in box_sizes:
            if box_size < self.min_box_size: break
            if self.progress_callback: self.progress_callback(f"计算盒子尺寸 {box_size:.4f}...")
            levels.append({'box_size': box_size, 'valid_count': self._count_total_for_iteration(trajectories, width, height, box_size)})
        valid_data = [(l['box_size'], l['valid_count']) for l in levels if l['valid_count'] > 0]
        if len(valid_data) < 2: return None, None, levels
        valid_box_sizes, valid_counts = zip(*valid_data)
        log_sizes, log_counts = np.log10(valid_box_sizes), np.log10(valid_counts)
        try: slope, intercept, r_value, _, _ = stats.linregress(log_sizes, log_counts)
        except (np.linalg.LinAlgError, ValueError): return None, None, levels
        fit_data = {'log_sizes': log_sizes, 'log_counts': log_counts, 'coeffs': [slope, intercept], 'fractal_intercept': intercept, 'fractal_n0': 10**intercept, 'r_squared': r_value**2}
        return -slope, fit_data, levels

class FractalAnalysisUtils:
    @staticmethod
    def _fit_from_levels_data(levels_data):
        if not levels_data: return None, None
        average_levels = [{'box_size': bs, 'valid_count': np.mean(cs)} for bs, cs in sorted(levels_data.items(), reverse=True)]
        filtered_average_levels = [l for l in average_levels if l['valid_count'] > 0]
        if len(filtered_average_levels) >= 2:
            log_sizes = [np.log10(l['box_size']) for l in filtered_average_levels]
            log_counts = [np.log10(l['valid_count']) for l in filtered_average_levels]
            try:
                if np.isinf(log_counts).any() or np.isnan(log_counts).any(): return None, None
                coeffs = np.polyfit(log_sizes, log_counts, 1)
                return -coeffs[0], 10 ** coeffs[1]
            except np.linalg.LinAlgError: return None, None
        return None, None

# --- FILE: fractal_analysis.py ---

# --- FILE: fractal_analysis.py ---

    @staticmethod
    def calculate_average_2d_dimension(fractures, cube_size, calc_iterations, gen_iterations, num_slices_per_direction, calc_slice_profile=True, calc_center_only=False):
        """
        计算平均二维分形维数
        :param calc_center_only: 如果为 True，仅计算 [0.4, 0.6] 范围内的切片，忽略边缘切片。
        """
        if not fractures: return {}
        
        all_levels_data, yoz_levels_data, xoz_levels_data, xoy_levels_data = {}, {}, {}, {}
        
        # 仅当需要计算分布时，初始化容器
        profile_levels_data = [{} for _ in range(num_slices_per_direction)] if calc_slice_profile else None
        
        calculator_2d = FractalDimension2DCalculator(iterations=calc_iterations, generation_iterations=gen_iterations)
        
        for i in range(num_slices_per_direction):
            # 1. 计算相对位置 (0.0 ~ 1.0)
            rel_pos = i * (1.0 / (num_slices_per_direction - 1)) if num_slices_per_direction > 1 else 0.5
            val = rel_pos * cube_size
            
            # 2. === 核心修改：中心区域过滤器 ===
            # 如果勾选了仅计算中心，且当前切片不在 [0.4, 0.6] 范围内，直接跳过
            if calc_center_only:
                if rel_pos < 0.4 - 1e-9 or rel_pos > 0.6 + 1e-9: # 加一点容差处理浮点数
                    continue
            # =================================
            
            current_slice_faces = [
                {'name': f'x={val:.3f}', 'direction': 'yoz'}, 
                {'name': f'y={val:.3f}', 'direction': 'xoz'}, 
                {'name': f'z={val:.3f}', 'direction': 'xoy'}
            ]
            
            for face_config in current_slice_faces:
                face_name, direction = face_config['name'], face_config['direction']
                face_line_segments = []
                
                for fracture in fractures:
                    if hasattr(fracture, 'boundary_lines'):
                        for line_info in fracture.boundary_lines:
                            if line_info['face_name'] == face_name:
                                points = line_info['points']
                                indices = [1, 2] if direction == 'yoz' else ([0, 2] if direction == 'xoz' else [0, 1])
                                if len(points) >= 2: 
                                    face_line_segments.append([(p[indices[0]], p[indices[1]]) for p in points])
                
                if face_line_segments:
                    # 只传递一个 cube_size 参数 (正方形切面)
                    _, _, levels = calculator_2d.calculate_fractal_dimension(
                        face_line_segments, cube_size
                    )
                    
                    if levels:
                        for level in levels:
                            box_size, count = level['box_size'], level['valid_count']
                            
                            # A. 累加到“标量计算容器”
                            # 因为跳过了边缘切片，这里累加的自然就全是中心数据了
                            # 所以最后算出来的 all_dim, yoz_dim 等自然就是中心域结果
                            all_levels_data.setdefault(box_size, []).append(count)
                            if direction == 'yoz': yoz_levels_data.setdefault(box_size, []).append(count)
                            elif direction == 'xoz': xoz_levels_data.setdefault(box_size, []).append(count)
                            elif direction == 'xoy': xoy_levels_data.setdefault(box_size, []).append(count)
                            
                            # B. 累加到“分布计算容器”
                            if calc_slice_profile and profile_levels_data is not None:
                                profile_levels_data[i].setdefault(box_size, []).append(count)

        # 3. 计算标量结果 (基于筛选后的数据)
        all_dim, all_n0 = FractalAnalysisUtils._fit_from_levels_data(all_levels_data)
        yoz_dim, yoz_n0 = FractalAnalysisUtils._fit_from_levels_data(yoz_levels_data)
        xoz_dim, xoz_n0 = FractalAnalysisUtils._fit_from_levels_data(xoz_levels_data)
        xoy_dim, xoy_n0 = FractalAnalysisUtils._fit_from_levels_data(xoy_levels_data)
        
        # 4. 计算分布列表
        slice_profile_dim = None
        slice_profile_n0 = None
        
        if calc_slice_profile and profile_levels_data:
            slice_profile_dim = []
            slice_profile_n0 = []
            for i in range(num_slices_per_direction):
                # 如果被过滤器跳过，profile_levels_data[i] 是空的
                # _fit_from_levels_data 对空数据会返回 None, None
                # 这样我们就得到一个中间有值，两头是 None 的列表
                p_dim, p_n0 = FractalAnalysisUtils._fit_from_levels_data(profile_levels_data[i])
                slice_profile_dim.append(p_dim)
                slice_profile_n0.append(p_n0)

        return {
            'all_dim': all_dim, 'all_n0': all_n0, 
            'yoz_dim': yoz_dim, 'yoz_n0': yoz_n0,
            'xoz_dim': xoz_dim, 'xoz_n0': xoz_n0, 
            'xoy_dim': xoy_dim, 'xoy_n0': xoy_n0,
            'slice_profile_dim': slice_profile_dim,
            'slice_profile_n0': slice_profile_n0
        }