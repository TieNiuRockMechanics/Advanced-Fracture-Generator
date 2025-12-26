import numpy as np
from datetime import datetime
# 假设 fracture_model.py 在同一路径下或可被导入
from fracture_model import EllipticalFracture


class FractureExporter:
    """椭圆裂缝网络导出器"""
    
    def __init__(self):
        pass
    
    def export_to_stl(self, fractures, cube_size, filename):
        """导出为STL格式"""
        triangles = []
        
        # 添加立方体边框（线框转为薄片）
        cube_triangles = self._generate_cube_wireframe_triangles(cube_size)
        triangles.extend(cube_triangles)
        
        # 添加椭圆裂缝面
        for fracture in fractures:
            # 使用裁切后的顶点，如果没有则使用原始顶点
            if fracture.clipped_vertices is not None and len(fracture.clipped_vertices) >= 3:
                vertices = fracture.clipped_vertices
            else:
                vertices = fracture.vertices
            
            if len(vertices) >= 3:
                fracture_triangles = self._triangulate_polygon(vertices)
                triangles.extend(fracture_triangles)
        
        # 写入STL文件
        self._write_stl_file(triangles, filename)
    
    def export_to_obj(self, fractures, cube_size, filename):
        """导出为OBJ格式"""
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        # 添加立方体顶点和面
        cube_vertices, cube_faces = self._generate_cube_mesh(cube_size)
        all_vertices.extend(cube_vertices)
        all_faces.extend([(f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset) for f in cube_faces])
        vertex_offset += len(cube_vertices)
        
        # 添加椭圆裂缝面
        for i, fracture in enumerate(fractures):
            # 使用裁切后的顶点，如果没有则使用原始顶点
            if fracture.clipped_vertices is not None and len(fracture.clipped_vertices) >= 3:
                vertices = fracture.clipped_vertices
            else:
                vertices = fracture.vertices
            
            if len(vertices) >= 3:
                fracture_triangles = self._triangulate_polygon(vertices)
                for triangle in fracture_triangles:
                    # 添加顶点
                    v1_idx = len(all_vertices)
                    v2_idx = len(all_vertices) + 1
                    v3_idx = len(all_vertices) + 2
                    
                    all_vertices.extend(triangle)
                    all_faces.append((v1_idx + 1, v2_idx + 1, v3_idx + 1))  # OBJ格式从1开始索引
        
        # 写入OBJ文件
        self._write_obj_file(all_vertices, all_faces, filename)
    
    def export_to_ply(self, fractures, cube_size, filename):
        """导出为PLY格式"""
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        # 添加立方体
        cube_vertices, cube_faces = self._generate_cube_mesh(cube_size)
        all_vertices.extend(cube_vertices)
        all_faces.extend([(f[0] + vertex_offset, f[1] + vertex_offset, f[2] + vertex_offset) for f in cube_faces])
        vertex_offset += len(cube_vertices)
        
        # 添加椭圆裂缝面
        for fracture in fractures:
            # 使用裁切后的顶点，如果没有则使用原始顶点
            if fracture.clipped_vertices is not None and len(fracture.clipped_vertices) >= 3:
                vertices = fracture.clipped_vertices
            else:
                vertices = fracture.vertices
            
            if len(vertices) >= 3:
                fracture_triangles = self._triangulate_polygon(vertices)
                for triangle in fracture_triangles:
                    v1_idx = len(all_vertices)
                    v2_idx = len(all_vertices) + 1
                    v3_idx = len(all_vertices) + 2
                    
                    all_vertices.extend(triangle)
                    all_faces.append((v1_idx, v2_idx, v3_idx))
        
        # 写入PLY文件
        self._write_ply_file(all_vertices, all_faces, filename)
    
    def export_fracture_data(self, fractures, cube_size, filename):
        """导出椭圆裂缝数据为文本格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# 椭圆裂缝网络数据\n")
            f.write(f"# 立方体尺寸: {cube_size}\n")
            f.write(f"# 裂缝数量: {len(fractures)}\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, fracture in enumerate(fractures):
                f.write(f"椭圆裂缝 {i+1}:\n")
                f.write(f"  中心位置: ({fracture.center[0]:.6f}, {fracture.center[1]:.6f}, {fracture.center[2]:.6f})\n")
                f.write(f"  法向量: ({fracture.normal[0]:.6f}, {fracture.normal[1]:.6f}, {fracture.normal[2]:.6f})\n")
                f.write(f"  长半轴: {fracture.semi_major_axis:.6f}\n")
                f.write(f"  短半轴: {fracture.semi_minor_axis:.6f}\n")
                f.write(f"  倾角: {np.rad2deg(fracture.inclination):.2f}度\n")
                f.write(f"  方位角: {np.rad2deg(fracture.azimuth):.2f}度\n")
                f.write(f"  面积: {fracture.area:.6f}\n")
                f.write(f"  迭代层级: {fracture.iteration_level}\n")
                f.write(f"  裂缝ID: {fracture.fracture_id}\n")
                f.write(f"  是否被裁切: {'是' if fracture.was_clipped else '否'}\n")
                
                # 使用裁切后的顶点，如果没有则使用原始顶点
                if fracture.clipped_vertices is not None and len(fracture.clipped_vertices) > 0:
                    vertices = fracture.clipped_vertices
                    f.write(f"  裁切后顶点数量: {len(vertices)}\n")
                    f.write(f"  裁切后顶点坐标:\n")
                else:
                    vertices = fracture.vertices
                    f.write(f"  原始顶点数量: {len(vertices)}\n")
                    f.write(f"  原始顶点坐标:\n")
                
                for j, vertex in enumerate(vertices):
                    f.write(f"    {j+1}: ({vertex[0]:.6f}, {vertex[1]:.6f}, {vertex[2]:.6f})\n")
                
                # 如果有边界线信息，也记录下来
                if hasattr(fracture, 'boundary_lines') and len(fracture.boundary_lines) > 0:
                    f.write(f"  边界线数量: {len(fracture.boundary_lines)}\n")
                    for k, line_info in enumerate(fracture.boundary_lines):
                        f.write(f"    边界线 {k+1} (面: {line_info['face_name']}):\n")
                        for l, point in enumerate(line_info['points']):
                            f.write(f"      点{l+1}: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})\n")
                
                f.write("\n")
    
    def _triangulate_polygon(self, vertices):
        """将多边形三角化"""
        if len(vertices) < 3:
            return []
        
        triangles = []
        # 简单的扇形三角剖分
        for i in range(1, len(vertices) - 1):
            triangle = [vertices[0], vertices[i], vertices[i + 1]]
            triangles.append(triangle)
        
        return triangles
    
    def _generate_cube_wireframe_triangles(self, size, thickness=0.01):
        """生成立方体线框的三角形（将线转为薄片）"""
        triangles = []
        
        # 立方体的8个顶点
        vertices = [
            [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
            [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]
        ]
        
        # 立方体的12条边
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
        ]
        
        # 为每条边生成薄片三角形
        for edge in edges:
            start = np.array(vertices[edge[0]])
            end = np.array(vertices[edge[1]])
            direction = end - start
            length = np.linalg.norm(direction)
            
            if length > 0:
                direction = direction / length
                # 找一个垂直方向
                if abs(direction[2]) < 0.9:
                    perp1 = np.cross(direction, [0, 0, 1])
                else:
                    perp1 = np.cross(direction, [1, 0, 0])
                perp1 = perp1 / np.linalg.norm(perp1) * thickness
                perp2 = np.cross(direction, perp1)
                perp2 = perp2 / np.linalg.norm(perp2) * thickness
                
                # 创建四个顶点的薄片
                v1 = start + perp1
                v2 = start - perp1
                v3 = end - perp1
                v4 = end + perp1
                
                # 两个三角形组成薄片
                triangles.append([v1, v2, v3])
                triangles.append([v1, v3, v4])
        
        return triangles
    
    def _generate_cube_mesh(self, size):
        """生成立方体网格"""
        vertices = [
            [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
            [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]
        ]
        
        faces = [
            # 底面
            [0, 1, 2], [0, 2, 3],
            # 顶面
            [4, 7, 6], [4, 6, 5],
            # 前面
            [0, 4, 5], [0, 5, 1],
            # 后面
            [2, 6, 7], [2, 7, 3],
            # 左面
            [0, 3, 7], [0, 7, 4],
            # 右面
            [1, 5, 6], [1, 6, 2]
        ]
        
        return vertices, faces
    
    def _write_stl_file(self, triangles, filename):
        """写入STL文件"""
        with open(filename, 'w') as f:
            f.write("solid elliptical_fracture_network\n")
            
            for triangle in triangles:
                # 计算法向量
                v1, v2, v3 = triangle
                normal = np.cross(v2 - v1, v3 - v1)
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                else:
                    normal = [0, 0, 1]
                
                f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                f.write("    outer loop\n")
                for vertex in triangle:
                    f.write(f"      vertex {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write("endsolid elliptical_fracture_network\n")
    
    def _write_obj_file(self, vertices, faces, filename):
        """写入OBJ文件"""
        with open(filename, 'w') as f:
            f.write("# 椭圆裂缝网络模型\n")
            f.write(f"# 顶点数: {len(vertices)}\n")
            f.write(f"# 面数: {len(faces)}\n\n")
            
            # 写入顶点
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            f.write("\n")
            
            # 写入面
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    def _write_ply_file(self, vertices, faces, filename):
        """写入PLY文件"""
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # 写入顶点
            for vertex in vertices:
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # 写入面
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
