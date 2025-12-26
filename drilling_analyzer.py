# --- START OF FILE drilling_analyzer.py ---

import numpy as np
from typing import List, Dict, Any, Optional

# 从项目中的其他模块导入必要的类
from random_manager import RandomStateManager
from fracture_model import EllipticalFracture


class DrillingAnalyzer:
    """
    钻孔分析器模块。

    该模块负责模拟垂直钻孔与三维裂缝网络的几何相交分析。
    其功能是独立的，不依赖于任何UI或绘图库。
    """

    def __init__(self, rng_manager: Optional[RandomStateManager] = None):
        """
        初始化钻孔分析器。

        Args:
            rng_manager (Optional[RandomStateManager]): 一个可选的随机状态管理器，
                用于提供可复现的随机数生成器。如果未提供，将使用系统时间创建
                一个临时的随机生成器。
        """
        self.random_point: Optional[np.ndarray] = None
        self.cube_size: float = 10.0
        self.intersected_fractures: List[Dict[str, Any]] = []

        if rng_manager:
            self.rng = rng_manager.drilling_rng
        else:
            # 如果没有提供管理器，则创建一个临时的，以保证模块可以独立运行
            import time
            self.rng = RandomStateManager(int(time.time() * 1000)).drilling_rng

    def generate_random_point(self, cube_size: float) -> np.ndarray:
        """
        在立方体的XY平面范围内生成一个随机的钻孔起始点。

        Args:
            cube_size (float): 立方体的边长。

        Returns:
            np.ndarray: 一个三维坐标数组 [x, y, 0]，表示钻孔的起始位置。
        """
        self.cube_size = cube_size
        x_coord = self.rng.uniform(0, cube_size)
        y_coord = self.rng.uniform(0, cube_size)
        self.random_point = np.array([x_coord, y_coord, 0])
        return self.random_point

    def analyze_drilling_probability(
        self,
        fractures: List[EllipticalFracture],
        cube_size: float,
        random_point: np.ndarray
    ) -> Dict[str, Any]:
        """
        分析单次垂直钻孔与裂缝网络的相交情况。
        这是该模块的核心功能。

        Args:
            fractures (List[EllipticalFracture]): 包含所有裂缝对象的列表。
            cube_size (float): 模拟立方体的边长。
            random_point (np.ndarray): 钻孔的起始点 [x, y, 0]。

        Returns:
            Dict[str, Any]: 一个包含分析结果的字典，包括钻孔线信息、
                            所有相交裂缝的详细信息以及相交总数。
        """
        self.intersected_fractures = []

        # 定义从 z=0 到 z=cube_size 的垂直钻孔线段
        line_start = random_point
        line_end = np.array([random_point[0], random_point[1], cube_size])

        # 遍历所有裂缝，检查是否与钻孔线相交
        for fracture in fractures:
            intersection_info = self._line_fracture_intersection(
                line_start, line_end, fracture
            )

            if intersection_info is not None:
                self.intersected_fractures.append(intersection_info)

        # 准备返回结果
        return {
            'random_point': random_point,
            'line_start': line_start,
            'line_end': line_end,
            'intersected_fractures': self.intersected_fractures,
            'total_intersections': len(self.intersected_fractures)
        }

    def _line_fracture_intersection(
        self,
        line_start: np.ndarray,
        line_end: np.ndarray,
        fracture: EllipticalFracture
    ) -> Optional[Dict[str, Any]]:
        """
        计算一条垂直线段与一个裂缝平面的交点。
        这是一个内部辅助方法。

        Args:
            line_start (np.ndarray): 线段起点。
            line_end (np.ndarray): 线段终点。
            fracture (EllipticalFracture): 要检查的裂缝对象。

        Returns:
            Optional[Dict[str, Any]]: 如果相交，则返回包含交点信息的字典；
                                      否则返回 None。
        """
        # 优先使用裁切后的顶点，如果不存在则使用原始顶点
        vertices = fracture.clipped_vertices if fracture.clipped_vertices is not None and len(fracture.clipped_vertices) >= 3 else fracture.vertices

        if len(vertices) < 3:
            return None

        # 使用裂缝对象中已经计算好的法向量
        normal = fracture.normal

        # 计算线-面相交
        line_direction = line_end - line_start
        denominator = np.dot(normal, line_direction)

        # 避免除以零（线与平面平行）
        if abs(denominator) < 1e-10:
            return None

        t = np.dot(normal, vertices[0] - line_start) / denominator

        # 交点必须在线段内部 (0 <= t <= 1)
        if not (0 <= t <= 1):
            return None

        intersection_point = line_start + t * line_direction

        # 检查交点是否在裂缝多边形内部
        if self._point_in_polygon_3d(intersection_point, vertices, normal):
            return {
                'fracture_id': fracture.fracture_id,
                'intersection_point': intersection_point,
                'z_coordinate': intersection_point[2],
                'fracture_area': fracture.area,
                'fracture': fracture  # 返回裂缝对象的引用，以便访问更多属性
            }

        return None

    def _point_in_polygon_3d(
        self,
        point: np.ndarray,
        vertices: np.ndarray,
        normal: np.ndarray
    ) -> bool:
        """
        检查一个点是否在一个三维凸多边形内部。
        通过将问题投影到二维平面来解决。

        Args:
            point (np.ndarray): 要检查的点。
            vertices (np.ndarray): 多边形的顶点数组。
            normal (np.ndarray): 多边形的法向量。

        Returns:
            bool: 如果点在多边形内，则为 True，否则为 False。
        """
        # 创建一个局部二维坐标系
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, [0, 0, 1])
        else:
            u = np.cross(normal, [1, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)

        # 将三维点投影到这个二维坐标系
        point_2d = np.array([np.dot(point - vertices[0], u), np.dot(point - vertices[0], v)])
        
        vertices_2d = []
        for vertex in vertices:
            vertex_rel = vertex - vertices[0]
            vertices_2d.append([np.dot(vertex_rel, u), np.dot(vertex_rel, v)])

        return self._point_in_polygon_2d(point_2d, np.array(vertices_2d))

    def _point_in_polygon_2d(
        self,
        point: np.ndarray,
        vertices: np.ndarray
    ) -> bool:
        """
        使用射线法（Ray Casting Algorithm）判断一个二维点是否在多边形内部。

        Args:
            point (np.ndarray): 要检查的二维点 [x, y]。
            vertices (np.ndarray): 多边形的二维顶点数组。

        Returns:
            bool: 如果点在多边形内，则为 True，否则为 False。
        """
        x, y = point
        n = len(vertices)
        inside = False

        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

# --- END OF FILE drilling_analyzer.py ---