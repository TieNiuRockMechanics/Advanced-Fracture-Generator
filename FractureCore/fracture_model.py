import numpy as np
import open3d as o3d
import math



class EllipticalFracture:
    """椭圆形裂缝面类 (功能增强版)"""
    def __init__(self, center, semi_major_axis, semi_minor_axis, 
                inclination=0, azimuth=0, iteration_level=0, fracture_id=0):
        self.center = np.array(center)
        self.semi_major_axis = semi_major_axis
        self.semi_minor_axis = semi_minor_axis
        self.inclination = inclination
        self.azimuth = azimuth
        self.iteration_level = iteration_level
        self.fracture_id = fracture_id
        self.area = np.pi * semi_major_axis * semi_minor_axis
        self.aspect_ratio = semi_major_axis / semi_minor_axis if semi_minor_axis > 0 else 1.0
        self.normal = self.get_actual_normal()
        self.vertices = self._generate_ellipse_vertices()
        self._area_cache = {}
        self._bounding_box = None

        # --- 新增属性 (从旧代码移植) ---
        self.clipped_vertices = None
        self.was_clipped = False
        self.boundary_lines = []
        # --- 移植结束 ---

    def get_bounding_box(self):
        if self._bounding_box is None:
            if self.vertices is not None and len(self.vertices) > 0:
                min_bound = np.min(self.vertices, axis=0)
                max_bound = np.max(self.vertices, axis=0)
                self._bounding_box = (min_bound, max_bound)
            else:
                return (self.center, self.center)
        return self._bounding_box

    def _generate_ellipse_vertices(self, num_points=32):
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        ellipse_points = np.array([
            [self.semi_major_axis * np.cos(angle), self.semi_minor_axis * np.sin(angle), 0] 
            for angle in angles
        ])
        transformed_points = self._transform_ellipse_points(ellipse_points)
        return self.center + transformed_points

    def _transform_ellipse_points(self, points):
        cos_inc, sin_inc = np.cos(self.inclination), np.sin(self.inclination)
        cos_az, sin_az = np.cos(self.azimuth), np.sin(self.azimuth)
        rotation_inc = np.array([[cos_inc, 0, sin_inc], [0, 1, 0], [-sin_inc, 0, cos_inc]])
        rotation_az = np.array([[cos_az, -sin_az, 0], [sin_az, cos_az, 0], [0, 0, 1]])
        combined_rotation = rotation_az @ rotation_inc
        return np.dot(points, combined_rotation.T)

    def get_area_in_box(self, box_min, box_max):
        cache_key = (tuple(box_min), tuple(box_max))
        if cache_key in self._area_cache:
            return self._area_cache[cache_key]
        
        if not self._quick_intersection_check(box_min, box_max):
            result = 0.0
        else:
            clipped_vertices = self._clip_ellipse_to_box(box_min, box_max)
            result = self._calculate_polygon_area_3d(clipped_vertices) if len(clipped_vertices) >= 3 else 0.0
        
        self._area_cache[cache_key] = result
        return result

    def _quick_intersection_check(self, box_min, box_max):
        center_to_box_min = np.maximum(box_min - self.center, 0)
        center_to_box_max = np.maximum(self.center - box_max, 0)
        distance_to_box = np.linalg.norm(center_to_box_min + center_to_box_max)
        return distance_to_box <= max(self.semi_major_axis, self.semi_minor_axis)

    def _clip_ellipse_to_box(self, box_min, box_max):
        clipped = self.vertices.copy()
        if len(clipped) == 0:
            return clipped
        
        planes = [
            (np.array([1, 0, 0]), np.array([box_min[0], 0, 0])), (np.array([-1, 0, 0]), np.array([box_max[0], 0, 0])),
            (np.array([0, 1, 0]), np.array([0, box_min[1], 0])), (np.array([0, -1, 0]), np.array([0, box_max[1], 0])),
            (np.array([0, 0, 1]), np.array([0, 0, box_min[2]])), (np.array([0, 0, -1]), np.array([0, 0, box_max[2]])),
        ]
        
        for plane_normal, plane_point in planes:
            clipped = self._clip_polygon_by_plane(clipped, plane_normal, plane_point)
            if len(clipped) == 0:
                break
        return clipped

    def _clip_polygon_by_plane(self, vertices, plane_normal, plane_point):
        if len(vertices) == 0: return vertices
        clipped_vertices = []
        for i in range(len(vertices)):
            current_vertex = vertices[i]
            next_vertex = vertices[(i + 1) % len(vertices)]
            current_dist = np.dot(current_vertex - plane_point, plane_normal)
            next_dist = np.dot(next_vertex - plane_point, plane_normal)
            
            if current_dist >= -1e-10:
                clipped_vertices.append(current_vertex)
                if next_dist < -1e-10:
                    intersection = self._line_plane_intersection(current_vertex, next_vertex, plane_normal, plane_point)
                    if intersection is not None: clipped_vertices.append(intersection)
            elif next_dist >= -1e-10:
                intersection = self._line_plane_intersection(current_vertex, next_vertex, plane_normal, plane_point)
                if intersection is not None: clipped_vertices.append(intersection)
        return np.array(clipped_vertices) if clipped_vertices else np.array([])

    def _line_plane_intersection(self, p1, p2, plane_normal, plane_point):
        direction = p2 - p1
        denominator = np.dot(direction, plane_normal)
        if abs(denominator) < 1e-10: return None
        t = np.dot(plane_point - p1, plane_normal) / denominator
        return p1 + t * direction if 0 <= t <= 1 else None

    def _calculate_polygon_area_3d(self, vertices):
        if len(vertices) < 3: return 0.0
        total_area = 0.0
        for i in range(1, len(vertices) - 1):
            v1 = vertices[i] - vertices[0]
            v2 = vertices[i + 1] - vertices[0]
            total_area += 0.5 * np.linalg.norm(np.cross(v1, v2))
        return total_area

    def get_actual_normal(self):
        initial_normal = np.array([0, 0, 1])
        cos_inc, sin_inc = np.cos(self.inclination), np.sin(self.inclination)
        cos_az, sin_az = np.cos(self.azimuth), np.sin(self.azimuth)
        rotation_inc = np.array([[cos_inc, 0, sin_inc], [0, 1, 0], [-sin_inc, 0, cos_inc]])
        rotation_az = np.array([[cos_az, -sin_az, 0], [sin_az, cos_az, 0], [0, 0, 1]])
        combined_rotation = rotation_az @ rotation_inc
        actual_normal = combined_rotation @ initial_normal
        return actual_normal / np.linalg.norm(actual_normal)

    # --- 新增方法 (从旧代码移植，用于可视化和2D分析) ---
    def clip_to_cube(self, cube_size, num_slices_per_direction):
        original_vertices = self.vertices.copy()
        self.clipped_vertices = self._clip_fracture_to_cube(original_vertices, cube_size)
        if len(self.clipped_vertices) != len(original_vertices):
            self.was_clipped = True
        else:
            self.was_clipped = not np.allclose(self.clipped_vertices, original_vertices, atol=1e-10)
        
        if len(self.clipped_vertices) >= 3:
            # 将参数传递下去
            self.boundary_lines = self._extract_boundary_lines_from_clipped(self.clipped_vertices, cube_size, num_slices_per_direction)
        return self.clipped_vertices

    def _clip_fracture_to_cube(self, vertices, cube_size):
        if len(vertices) == 0: return vertices
        clipped = vertices.copy()
        planes = [
            (np.array([1, 0, 0]), np.array([0, 0, 0])), (np.array([-1, 0, 0]), np.array([cube_size, 0, 0])),
            (np.array([0, 1, 0]), np.array([0, 0, 0])), (np.array([0, -1, 0]), np.array([0, cube_size, 0])),
            (np.array([0, 0, 1]), np.array([0, 0, 0])), (np.array([0, 0, -1]), np.array([0, 0, cube_size])),
        ]
        for plane_normal, plane_point in planes:
            clipped = self._clip_polygon_by_plane(clipped, plane_normal, plane_point)
            if len(clipped) == 0: break
        return clipped

    def _extract_boundary_lines_from_clipped(self, clipped_vertices, cube_size, num_slices_per_direction):
        if len(clipped_vertices) < 3: return []

        min_bound, max_bound = self.get_bounding_box()

        boundary_lines, tolerance = [], 1e-6
        faces = []
        for i in range(num_slices_per_direction): # <--- 使用参数
            # <--- 使用参数
            val = i * (1.0 / (num_slices_per_direction - 1)) * cube_size if num_slices_per_direction > 1 else 0.5 * cube_size
            faces.append({'coord': 0, 'value': val, 'name': f'x={val:.3f}'})
            faces.append({'coord': 1, 'value': val, 'name': f'y={val:.3f}'})
            faces.append({'coord': 2, 'value': val, 'name': f'z={val:.3f}'})
        
        for face in faces:
            # --- 新增优化：边界盒快速筛选 ---
            coord = face['coord']
            value = face['value']
            
            # 如果裂缝的整个包围盒都在切片平面的同一侧，则不可能相交，直接跳过。
            if max_bound[coord] < value - tolerance or min_bound[coord] > value + tolerance:
                continue
            # --- 优化结束 ---
            
            # 只有通过了快速筛选的平面，才进行昂贵的逐顶点交线计算
            intersection_points = self._calculate_polygon_plane_intersection(clipped_vertices, face['coord'], face['value'], tolerance)
            if len(intersection_points) >= 2:
                boundary_lines.append({
                    'points': np.array(intersection_points), 'face_name': face['name'],
                    'face_coord': face['coord'], 'face_value': face['value']
                })
        return boundary_lines

    def _calculate_polygon_plane_intersection(self, vertices, coord, plane_value, tolerance):
        """
        计算多边形与平面的交线 (NumPy向量化优化版)。
        该方法通过数组操作替代Python循环，显著提升计算性能。
        """
        if len(vertices) < 3:
            return []

        # 1. 准备数据：创建所有边的起点(v1s)和终点(v2s)数组
        # np.roll(vertices, -1, axis=0) 将数组向上滚动一行，快速得到所有终点
        v1s = vertices
        v2s = np.roll(vertices, -1, axis=0)

        # 2. 计算第一类交点：顶点本身就在平面上
        # 使用布尔索引一次性找出所有符合条件的顶点
        on_plane_mask = np.abs(v1s[:, coord] - plane_value) < tolerance
        points_on_plane = v1s[on_plane_mask]

        # 3. 计算第二类交点：边与平面相交
        # 3a. 获取所有起点和终点在指定坐标轴上的值
        c1 = v1s[:, coord]
        c2 = v2s[:, coord]
        
        # 3b. 计算它们与平面的符号距离
        d1 = c1 - plane_value
        d2 = c2 - plane_value

        # 3c. 找出所有跨越平面的边 (d1*d2 < 0)
        #    同时，为了避免除以零，确保边的两个端点在该坐标轴上不重合
        denominator = c2 - c1
        crossing_mask = (d1 * d2 < 0) & (np.abs(denominator) > 1e-9)
        
        # 如果没有边跨越平面，则提前返回
        if not np.any(crossing_mask):
            intersection_points_crossing = np.array([])
        else:
            # 3d. 仅对跨越平面的边进行计算
            crossing_v1s = v1s[crossing_mask]
            crossing_d1 = d1[crossing_mask]
            crossing_denominator = denominator[crossing_mask]
            
            # 3e. 向量化计算所有交点的插值系数 t
            # t = (plane_value - c1) / (c2 - c1) = -d1 / (c2 - c1)
            t = -crossing_d1 / crossing_denominator
            
            # 3f. 向量化计算所有交点的三维坐标
            # t[:, np.newaxis] 将一维的t数组(n,)变为(n,1)，使其能与(n,3)的数组进行广播运算
            intersection_points_crossing = crossing_v1s + t[:, np.newaxis] * (v2s[crossing_mask] - crossing_v1s)

        # 4. 合并与去重
        # 4a. 将两类交点合并到一个数组中
        if intersection_points_crossing.shape[0] > 0:
            all_points = np.vstack([points_on_plane, intersection_points_crossing])
        else:
            all_points = points_on_plane
        
        if len(all_points) == 0:
            return []

        # 4b. 使用高效的去重方法：通过四舍五入到指定精度，然后用np.unique
        # 这里的 decimals=6 约等于 tolerance=1e-6
        unique_points = np.unique(all_points.round(decimals=6), axis=0)
        
        return list(unique_points)
    # --- 移植结束 ---


# 文件: fracture_model.py
# 请用此代码块完整替换 FractalBasedFractureGenerator 类

class FractalBasedFractureGenerator:
    """基于分形维数理论的裂缝生成器 (功能增强版)"""
    def __init__(self):
        self.cube_size = 1.0; self.fractal_dimension = 2.5; self.n0 = 1.0; self.max_iterations = 3
        self.aspect_ratio_base = 1.0; self.aspect_ratio_variation = 0.0
        self.inclination_base = 0.0; self.inclination_variation = np.pi
        self.azimuth_base = 0.0; self.azimuth_variation = np.pi
        self.random_aspect_ratio = True; self.random_inclination = True; self.random_azimuth = True
        self.use_advanced_model = False # 新增：Advanced Model 开关
        self.fractures_by_level = {}
        # 最终版：只有一个统一的、结构化的日志
        self.structured_log = []
        self.rng = np.random.default_rng()

    def _probabilistic_round(self, value):
        floor_value, decimal_part = int(value), value - int(value)
        return floor_value + 1 if self.rng.random() < decimal_part else floor_value

    def _generate_fisher_normal(self, mean_direction, kappa):
        """
        根据指定的平均方向和集中度kappa，生成一个服从Fisher分布的随机单位向量。
        """
        # 情况一：kappa为0，生成完全均匀的各向同性分布
        if kappa == 0:
            # 生成三个标准正态分布的随机数
            vec = self.rng.normal(size=3)
            norm = np.linalg.norm(vec)
            # 归一化，如果向量长度为0则返回一个默认向量
            return vec / norm if norm > 1e-9 else np.array([0, 0, 1])

        # 情况二：kappa > 0，生成聚集的各向异性分布
        # 1. 计算平均方向（必须是单位向量）
        mu = mean_direction / np.linalg.norm(mean_direction)
        
        # 2. 从 [0, 2*pi] 均匀采样一个角度 beta
        beta = self.rng.uniform(0, 2 * np.pi)
        
        # 3. 根据 kappa 采样角度 alpha (与平均方向的夹角)
        # 这是标准的Fisher分布采样公式
        w = math.exp(-2 * kappa)
        alpha = np.arccos(1 + (1 / kappa) * np.log(w + (1 - w) * self.rng.random()))
        
        # 4. 构建一个与 mu 正交的局部坐标系
        # 找到一个不与 mu 平行的向量 v_prime
        v_prime = self.rng.normal(size=3)
        while np.linalg.norm(np.cross(mu, v_prime)) < 1e-6:
             v_prime = self.rng.normal(size=3)
        
        # 使用 Gram-Schmidt 方法创建正交基
        u1 = np.cross(mu, v_prime)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(mu, u1)
        
        # 5. 在局部坐标系中旋转得到最终向量
        # 围绕 mu 旋转 beta，然后倾斜 alpha
        rotated_vec = (u1 * np.cos(beta) + u2 * np.sin(beta)) * np.sin(alpha) + mu * np.cos(alpha)
        
        return rotated_vec



    # 在 FractalBasedFractureGenerator 类中替换 generate_fractures 方法
    def generate_fractures(self, cube_size, fractal_dimension, n0, max_iterations=2, rng_manager=None, stop_event=None, is_isotropic=True, mean_inclination=90.0, mean_azimuth=0.0, kappa=0, **kwargs):
        if rng_manager:
            self.rng = rng_manager.fracture_rng
        
        # --- 核心修复：显式保存产状参数到 self ---
        # 1. 优先使用 kwargs 中的 main_app 传递过来的特定键名
        #    注意：main_app 传递的键名是 'orientation_isotropic', 'mean_inclination_deg', 'mean_azimuth_deg'
        #    如果这些键不存在，则回退到函数参数默认值。
        
        self.is_isotropic = kwargs.get('orientation_isotropic', is_isotropic)
        self.mean_inclination = kwargs.get('mean_inclination_deg', mean_inclination)
        self.mean_azimuth = kwargs.get('mean_azimuth_deg', mean_azimuth)
        self.kappa = kwargs.get('kappa', kappa)
        self.use_advanced_model = kwargs.get('use_advanced_model', False) # 确保高级模型标记也能被设置

        # 2. 处理其他常规参数 (形态参数等)
        #    依然保留 hasattr 检查以防止注入未知属性，但允许上述核心属性已被设置
        for key, value in kwargs.items():
            if hasattr(self, key): 
                setattr(self, key, value)
        
        self.cube_size, self.fractal_dimension, self.n0, self.max_iterations = cube_size, fractal_dimension, n0, max_iterations
        self.fractures_by_level = {}
        self.structured_log = []
        
        self._generate_statistics_header()
        
        self.structured_log.append(('log_start_generation', {'L': cube_size, 'D': fractal_dimension, 'N0': n0}))
        
        # 在每个耗时操作前检查停止事件
        if stop_event and stop_event.is_set():
            print("生成器在初始阶段被中止。")
            return self.fractures_by_level, self.structured_log

        self._generate_initial_fractures()

        for iteration in range(1, max_iterations + 1):
            if stop_event and stop_event.is_set():
                print(f"生成器在第 {iteration} 次迭代前被中止。")
                break # 如果需要停止，就跳出循环
            self._generate_iteration_fractures(iteration)
        
        self._generate_statistics_summary()
        
        return self.fractures_by_level, self.structured_log

    def _generate_initial_fractures(self):
        theoretical_count = self.n0 * (self.cube_size ** (-self.fractal_dimension))
        actual_count = self._probabilistic_round(theoretical_count)
        
        self.structured_log.append(('log_initial_state', {'L': self.cube_size}))
        self.structured_log.append(('log_initial_theory', {'theory_count': theoretical_count, 'actual_count': actual_count}))
        
        fractures, successful_count = [], 0
        # 重要修改，缩小生成裂缝的面积的区间范围
        min_L = self.cube_size * 1.1
        max_L = self.cube_size * 2 * 0.9
        min_area = round(min_L ** 2, 6)
        max_area = round(max_L ** 2, 6)
        
        for i in range(actual_count):
            fracture = self._generate_initial_elliptical_fracture(min_area, max_area, 0, i)
            if fracture: fractures.append(fracture); successful_count += 1
        self.fractures_by_level[0] = fractures
        
        self.structured_log.append(('log_success_generation', {'count': successful_count}))

    # ##########################################################################
    # ### START OF MODIFICATION ###
    # ##########################################################################
    # 此函数已根据 feedback control 方案完全重写
    def _generate_iteration_fractures(self, iteration):
        """
        【已修改】为指定迭代层级生成补充裂缝，采用反馈控制逻辑。
        """
        box_size = round(self.cube_size / (2 ** iteration), 6)
        theoretical_count = self.n0 * (box_size ** (-self.fractal_dimension))
        theoretical_count_rounded = self._probabilistic_round(theoretical_count)
        boxes = self._generate_boxes_for_iteration(iteration)
        
        all_fractures = self.get_fractures_up_to_level(iteration - 1)
        if all_fractures:
            fracture_bboxes = [frac.get_bounding_box() for frac in all_fractures]
            fracture_centers = np.array([(bbox[0] + bbox[1]) / 2.0 for bbox in fracture_bboxes])
            pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(fracture_centers)
            kdtree = o3d.geometry.KDTreeFlann(pcd)
            inherited_count = self._count_inherited_effective_fractures_optimized(boxes, box_size, kdtree, all_fractures, fracture_bboxes)
        else:
            inherited_count = 0

        need_to_add = max(0, theoretical_count_rounded - inherited_count)
        
        self.structured_log.append(('log_iteration_state', {'iteration': iteration, 'L': box_size}))
        self.structured_log.append(('log_iteration_theory', {'theory_count': theoretical_count, 'actual_count': theoretical_count_rounded}))
        self.structured_log.append(('log_iteration_supplement', {'inherited_count': inherited_count, 'needed': need_to_add}))
        
        # --- 新的反馈控制逻辑 ---
        new_fractures = []
        remaining_count_to_add = need_to_add
        
        if remaining_count_to_add > 0:
            # 定义裂缝尺寸和阈值
            min_L = box_size * 1.1
            max_L = box_size * 2 * 0.9
            min_area = round(min_L ** 2, 6)
            max_area = round(max_L ** 2, 6)
            threshold_area = round(box_size ** 2, 6)

            # 准备盒子搜索优化
            num_boxes = len(boxes)
            use_optimized_search = num_boxes > 125
            if use_optimized_search:
                grid_count = int(round(num_boxes**(1./3.)))

            # 循环直到补充的count满足要求
            while remaining_count_to_add > 0:
                was_fracture_added_this_pass = False
                # 尝试最多3000次来生成一条有效裂缝
                for _ in range(3000):
                    center = self.rng.uniform(0, self.cube_size, 3)
                    area = self.rng.uniform(min_area, max_area)
                    
                    # 使用当前已生成裂缝数作为ID
                    fracture = self._create_fracture(center, area, iteration, len(new_fractures))
                    
                    # 检查这条新裂缝满足了多少个盒子
                    boxes_to_check = []
                    if use_optimized_search:
                        clipped_center = np.clip(fracture.center, 0, self.cube_size - 1e-9)
                        center_indices = (clipped_center // box_size).astype(int)
                        i_center, j_center, k_center = center_indices
                        for i in range(max(0, i_center - 2), min(grid_count, i_center + 3)):
                            for j in range(max(0, j_center - 2), min(grid_count, j_center + 3)):
                                for k in range(max(0, k_center - 2), min(grid_count, k_center + 3)):
                                    box_index = i * (grid_count ** 2) + j * grid_count + k
                                    boxes_to_check.append(boxes[box_index])
                    else:
                        boxes_to_check = boxes
                    
                    satisfied_box_count = 0
                    for box in boxes_to_check:
                        if fracture.get_area_in_box(box['min'], box['max']) >= threshold_area:
                            satisfied_box_count += 1
                    
                    # 只要裂缝有效（满足至少一个盒子），就接受它
                    if satisfied_box_count > 0:
                        new_fractures.append(fracture)
                        remaining_count_to_add -= satisfied_box_count
                        was_fracture_added_this_pass = True
                        break # 成功，跳出3000次尝试循环
                
                # 死循环保护
                if not was_fracture_added_this_pass:
                    print(f"警告: 在第 {iteration} 次迭代中，3000次尝试后未能生成有效裂缝。")
                    print(f"  剩余待补充的count: {remaining_count_to_add}")
                    break # 跳出 while 循环

        # ==================== START: 按新格式输出 ====================
        print(f"迭代次数: {iteration} | "
              f"理论count: {theoretical_count_rounded} | "
              f"继承count: {inherited_count} | "
              f"补充count: {need_to_add} | "
              f"实际生成裂缝数: {len(new_fractures)}")
        # ===================== END: 按新格式输出 =====================
        
        self.fractures_by_level[iteration] = new_fractures
        self.structured_log.append(('log_success_generation_supplement', {'count': len(new_fractures)}))
    # ##########################################################################
    # ### END OF MODIFICATION ###
    # ##########################################################################

    # ... (其他 _generate... _count... _create... 函数保持不变) ...
    def _generate_boxes_for_iteration(self, iteration):
        boxes, box_size, grid_count = [], round(self.cube_size / (2 ** iteration), 6), 2 ** iteration
        for i, j, k in np.ndindex(grid_count, grid_count, grid_count):
            min_coords = np.round(np.array([i, j, k]) * box_size, 6)
            max_coords = np.round(np.array([i + 1, j + 1, k + 1]) * box_size, 6)
            boxes.append({'min': min_coords, 'max': max_coords})
        return boxes

    def _count_inherited_effective_fractures_optimized(self, boxes, box_size, kdtree, all_fractures, fracture_bboxes):
        threshold_area = round(box_size ** 2, 6)
        total_count = 0
        fracture_half_diagonals = [np.linalg.norm(bbox[1] - bbox[0]) / 2.0 for bbox in fracture_bboxes]
        max_fracture_half_diagonal = max(fracture_half_diagonals) if fracture_half_diagonals else 0
        for box in boxes:
            box_min, box_max = box['min'], box['max']
            box_center = (box_min + box_max) / 2.0
            box_half_diagonal = np.linalg.norm(box_max - box_min) / 2.0
            search_radius = box_half_diagonal + max_fracture_half_diagonal
            [k, idx_list, _] = kdtree.search_radius_vector_3d(box_center, search_radius)
            for frac_idx in idx_list:
                frac_bbox_min, frac_bbox_max = fracture_bboxes[frac_idx]
                if not (np.all(frac_bbox_min < box_max) and np.all(frac_bbox_max > box_min)): continue
                if all_fractures[frac_idx].get_area_in_box(box_min, box_max) >= threshold_area:
                    total_count += 1
        return total_count

# 文件: fracture_model.py (在 FractalBasedFractureGenerator 类中)

    def _create_fracture(self, center, area, iteration_level, fracture_id):
        """
        【已重构】创建单个裂缝对象。
        先处理形状（长短轴），再用Fisher分布处理产状（法向量）。
        """
        # 步骤 1: (保留) 处理长短轴比例，决定裂缝形状
        if self.random_aspect_ratio: 
            aspect_ratio = self.rng.uniform(1.0, 4.0)
        else: 
            aspect_ratio = max(1.0, self.rng.normal(self.aspect_ratio_base, self.aspect_ratio_variation))
        
        semi_minor_axis = np.sqrt(area / (np.pi * aspect_ratio))
        semi_major_axis = semi_minor_axis * aspect_ratio

        # 步骤 2: (升级) 处理产状，决定裂缝空间姿态
        
        # --- Advanced Model 逻辑分支 ---
        if getattr(self, 'use_advanced_model', False):
            # 1. 倾角 (Inclination): 截断正态分布 N(52.8, 17.6) clipped to [0, 90]
            # 使用简单的拒绝采样法 (Rejection Sampling)
            mu, sigma = 52.8, 17.6
            lower, upper = 0.0, 90.0
            while True:
                inc_deg = self.rng.normal(mu, sigma)
                if lower <= inc_deg <= upper:
                    break
            inclination_rad = np.deg2rad(inc_deg)

            # 2. 方位角 (Azimuth): 均匀分布 U(0, 360)
            az_deg = self.rng.uniform(0.0, 360.0)
            azimuth_rad = np.deg2rad(az_deg)

        else:
            # --- 原有逻辑 (Fisher 分布 / 各向同性) ---
            # is_isotropic, mean_inclination, mean_azimuth, kappa 将从kwargs传入
            is_isotropic = getattr(self, 'is_isotropic', True)
            kappa = 0 if is_isotropic else getattr(self, 'kappa', 0)

            # 2a. 将UI输入的平均方向（角度）转换为三维单位向量
            mean_inclination_rad = np.deg2rad(getattr(self, 'mean_inclination', 90.0))
            mean_azimuth_rad = np.deg2rad(getattr(self, 'mean_azimuth', 0.0))
            
            mean_direction_vector = np.array([
                np.sin(mean_inclination_rad) * np.cos(mean_azimuth_rad),
                np.sin(mean_inclination_rad) * np.sin(mean_azimuth_rad),
                np.cos(mean_inclination_rad)
            ])
            
            # 2b. 调用Fisher分布采样，生成最终的随机法向量
            final_normal = self._generate_fisher_normal(mean_direction_vector, kappa)

            # 步骤 3: (兼容) 从生成的法向量反向计算倾角和方位角，用于描述和实例化
            # 倾角是法向量与Z轴的夹角 (0, pi)
            inclination_rad = np.arccos(np.clip(final_normal[2], -1.0, 1.0))
            
            # 方位角是法向量在XY平面的投影与X轴的夹角 (0, 2*pi)
            azimuth_rad = np.arctan2(final_normal[1], final_normal[0])
            if azimuth_rad < 0:
                azimuth_rad += 2 * np.pi

        # 步骤 4: 实例化裂缝对象
        return EllipticalFracture(center, semi_major_axis, semi_minor_axis, 
                                inclination_rad, azimuth_rad, iteration_level, fracture_id)

    def _generate_initial_elliptical_fracture(self, min_area, max_area, iteration_level, fracture_id):
        threshold_area = round(self.cube_size ** 2, 6)
        for _ in range(3000):
            center = self.rng.uniform(-self.cube_size * 0.5, self.cube_size * 1.5, 3)
            area = self.rng.uniform(min_area, max_area)
            fracture = self._create_fracture(center, area, iteration_level, fracture_id)
            if fracture.get_area_in_box(np.array([0,0,0]), np.array([self.cube_size]*3)) >= threshold_area:
                return fracture
        return None

    def get_fractures_up_to_level(self, max_level):
        return [f for level in range(max_level + 1) if level in self.fractures_by_level for f in self.fractures_by_level[level]]

    def _generate_statistics_header(self):
        """生成统计信息的头部"""
        self.structured_log.insert(0, ('header', {'title': '生成过程详细日志'}))
        self.structured_log.insert(0, ('separator', {}))
        self.structured_log.insert(0, ('param', {'name': 'param_iterations', 'value': self.max_iterations}))
        self.structured_log.insert(0, ('param', {'name': 'param_n0', 'value': f"{self.n0:.3f}"}))
        self.structured_log.insert(0, ('param', {'name': 'param_dim', 'value': f"{self.fractal_dimension:.3f}"}))
        self.structured_log.insert(0, ('param', {'name': 'param_cube_size', 'value': f"{self.cube_size:.3f}"}))
        self.structured_log.insert(0, ('params_header', {}))
        self.structured_log.insert(0, ('header', {'title': '分形裂缝生成详细统计'}))

    def _generate_statistics_summary(self):
        """生成统计信息的总结部分"""
        summary_list = []
        total_fractures = 0
        for level, fractures in self.fractures_by_level.items():
            count = len(fractures)
            total_fractures += count
            if level == 0:
                summary_list.append(('stats_initial', {'count': count}))
            else:
                summary_list.append(('stats_iteration', {'level': level, 'count': count}))
        
        summary_list.append(('stats_total', {'count': total_fractures}))
        
        # 将总结信息插入到参数和详细日志之间
        # 参数部分的固定长度是 7
        self.structured_log[7:7] = summary_list
        self.structured_log.insert(7, ('separator', {}))
