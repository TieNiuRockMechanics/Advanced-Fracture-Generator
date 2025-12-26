# --- START OF FILE data_manager.py ---

import os
import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
import time

class SQLiteBackupManager:
    """
    SQLite备份管理器 (v2.2 - 支持展开分析数据并兼容旧数据库)
    """
    def __init__(self, backup_file_path=None, calculation_params=None):
        if backup_file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(os.getcwd(), "FD-Advanced_3D_temp_data")
            os.makedirs(backup_dir, exist_ok=True)
            backup_file_path = os.path.join(backup_dir, f"fractal_calculation_backup_{timestamp}.db")

        self.backup_file_path = backup_file_path
        self.calculation_params = calculation_params or {}
        self._initialize_backup_database()
        print(f"创建/连接SQLite备份数据库: {self.backup_file_path}")


# --- FILE: data_manager.py ---

    def _initialize_backup_database(self):
        """初始化SQLite备份数据库 (增加迁移逻辑以兼容旧版本)"""
        conn = sqlite3.connect(self.backup_file_path)
        try:
            conn.execute('PRAGMA journal_mode=WAL'); conn.execute('PRAGMA busy_timeout=30000')
            cursor = conn.cursor()
            
            # --- 步骤 1: 创建或更新 metadata 表 (不变) ---
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY,
                    start_time TEXT,
                    total_tasks INTEGER DEFAULT 0,
                    completed_tasks INTEGER DEFAULT 0,
                    last_update TEXT,
                    version TEXT DEFAULT '2.1',
                    calculation_params TEXT
                )
            ''')

            # --- 步骤 2: 创建 results 表 (不变) ---
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cube_size REAL, theoretical_dim REAL, theoretical_n0 REAL,
                    actual_3d_dim REAL, actual_3d_n0 REAL,
                    average_2d_dim_all REAL, average_2d_n0_all REAL,
                    average_2d_dim_yoz REAL, average_2d_n0_yoz REAL,
                    average_2d_dim_xoz REAL, average_2d_n0_xoz REAL,
                    average_2d_dim_xoy REAL, average_2d_n0_xoy REAL,
                    repeat_index INTEGER, success BOOLEAN, timestamp TEXT,
                    process_id INTEGER, task_info TEXT
                )
            ''')

            # --- 步骤 3: 【核心修改】数据库迁移逻辑 ---
            cursor.execute("PRAGMA table_info(results)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # 定义所有需要动态添加的新列
            new_columns = {
                'cuboid_unfolding_dim': 'REAL',
                'cuboid_unfolding_n0': 'REAL',
                'cylinder_unfolding_dim': 'REAL',
                'cylinder_unfolding_n0': 'REAL',
                # === 新增：用于存储切片趋势分布的 JSON 字符串 ===
                'slice_profile_dim': 'TEXT', 
                'slice_profile_n0': 'TEXT'
            }
            
            for col_name, col_type in new_columns.items():
                if col_name not in columns:
                    print(f"正在升级数据库：添加列 '{col_name}'...")
                    cursor.execute(f"ALTER TABLE results ADD COLUMN {col_name} {col_type}")

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_params ON results(cube_size, theoretical_dim, theoretical_n0)')
            cursor.execute('SELECT COUNT(*) FROM metadata')
            if cursor.fetchone()[0] == 0:
                now = datetime.now().isoformat()
                cursor.execute('''
                    INSERT INTO metadata (start_time, calculation_params, version, last_update)
                    VALUES (?, ?, '2.3', ?)
                ''', (now, json.dumps(self.calculation_params, ensure_ascii=False), now))

            conn.commit()
        finally:
            conn.close()

    def append_result(self, result_data, task_info=None):
        """追加单个计算结果到SQLite数据库"""
        max_retries = 5
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.backup_file_path, timeout=60.0)
                try:
                    conn.execute('PRAGMA journal_mode=WAL')
                    conn.execute('PRAGMA busy_timeout=30000')
                    cursor = conn.cursor()

                    timestamp = datetime.now().isoformat()
                    process_id = os.getpid()
                    task_info_json = json.dumps(task_info, ensure_ascii=False) if task_info else None

                    # 确保列表数据被转换为 JSON 字符串存储
                    slice_profile_dim_str = json.dumps(result_data.get('slice_profile_dim')) if result_data.get('slice_profile_dim') is not None else None
                    slice_profile_n0_str = json.dumps(result_data.get('slice_profile_n0')) if result_data.get('slice_profile_n0') is not None else None

                    conn.execute('BEGIN IMMEDIATE')
                    
                    cursor.execute('''
                        INSERT INTO results (
                            cube_size, theoretical_dim, theoretical_n0, actual_3d_dim, actual_3d_n0,
                            average_2d_dim_all, average_2d_n0_all,
                            average_2d_dim_yoz, average_2d_n0_yoz,
                            average_2d_dim_xoz, average_2d_n0_xoz,
                            average_2d_dim_xoy, average_2d_n0_xoy,
                            cuboid_unfolding_dim, cuboid_unfolding_n0,
                            cylinder_unfolding_dim, cylinder_unfolding_n0,
                            slice_profile_dim, slice_profile_n0,
                            repeat_index, success, timestamp, process_id, task_info
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result_data.get('cube_size'), result_data.get('theoretical_dim'), result_data.get('theoretical_n0'),
                        result_data.get('actual_3d_dim'), result_data.get('actual_3d_n0'),
                        result_data.get('average_2d_dim_all'), result_data.get('average_2d_n0_all'),
                        result_data.get('average_2d_dim_yoz'), result_data.get('average_2d_n0_yoz'),
                        result_data.get('average_2d_dim_xoz'), result_data.get('average_2d_n0_xoz'),
                        result_data.get('average_2d_dim_xoy'), result_data.get('average_2d_n0_xoy'),
                        result_data.get('cuboid_unfolding_dim'), result_data.get('cuboid_unfolding_n0'),
                        result_data.get('cylinder_unfolding_dim'), result_data.get('cylinder_unfolding_n0'),
                        # 新增数据 (JSON Strings)
                        slice_profile_dim_str, slice_profile_n0_str,
                        #
                        result_data.get('repeat_index'), result_data.get('success', True),
                        timestamp, process_id, task_info_json
                    ))

                    cursor.execute('SELECT COUNT(*) FROM results')
                    completed_count = cursor.fetchone()[0]
                    cursor.execute('UPDATE metadata SET completed_tasks = ?, last_update = ? WHERE id = 1', (completed_count, timestamp))
                    
                    conn.commit()
                    return True
                finally:
                    conn.close()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                else:
                    return False
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return False
        return False

    def load_backup_data(self):
        """从SQLite数据库加载备份数据 (兼容旧版本)"""
        try:
            if not os.path.exists(self.backup_file_path):
                return None
            
            conn = sqlite3.connect(self.backup_file_path, timeout=30.0)
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM metadata WHERE id = 1')
                metadata_row = cursor.fetchone()

                if not metadata_row:
                    return None
                
                try:
                    calculation_params = json.loads(metadata_row[6]) if metadata_row[6] and isinstance(metadata_row[6], str) else {}
                except (json.JSONDecodeError, TypeError):
                    calculation_params = {}

                metadata = {
                    'start_time': metadata_row[1], 'total_tasks': metadata_row[2], 'completed_tasks': metadata_row[3],
                    'last_update': metadata_row[4], 'version': metadata_row[5], 'calculation_params': calculation_params
                }
                
                # --- 核心修改：使用列名来安全地读取数据 ---
                cursor.execute('PRAGMA table_info(results)')
                column_info = cursor.fetchall()
                column_names = [info[1] for info in column_info]

                cursor.execute('SELECT * FROM results ORDER BY id')
                result_rows = cursor.fetchall()
                results = []
                for row in result_rows:
                    # 将每一行数据与列名打包成一个字典
                    row_dict = dict(zip(column_names, row))
                    
                    # 尝试解析 JSON 字符串回列表
                    try:
                        slice_profile_dim = json.loads(row_dict.get('slice_profile_dim')) if row_dict.get('slice_profile_dim') else None
                        slice_profile_n0 = json.loads(row_dict.get('slice_profile_n0')) if row_dict.get('slice_profile_n0') else None
                    except:
                        slice_profile_dim = None
                        slice_profile_n0 = None

                    # 使用 .get() 方法安全地从字典中取值
                    results.append({
                        'cube_size': row_dict.get('cube_size'), 'theoretical_dim': row_dict.get('theoretical_dim'),
                        'theoretical_n0': row_dict.get('theoretical_n0'), 'actual_3d_dim': row_dict.get('actual_3d_dim'),
                        'actual_3d_n0': row_dict.get('actual_3d_n0'),
                        'average_2d_dim_all': row_dict.get('average_2d_dim_all'), 'average_2d_n0_all': row_dict.get('average_2d_n0_all'),
                        'average_2d_dim_yoz': row_dict.get('average_2d_dim_yoz'), 'average_2d_n0_yoz': row_dict.get('average_2d_n0_yoz'),
                        'average_2d_dim_xoz': row_dict.get('average_2d_dim_xoz'), 'average_2d_n0_xoz': row_dict.get('average_2d_n0_xoz'),
                        'average_2d_dim_xoy': row_dict.get('average_2d_dim_xoy'), 'average_2d_n0_xoy': row_dict.get('average_2d_n0_xoy'),
                        # 新增数据，如果旧DB中不存在这些列，.get() 会安全返回 None
                        'cuboid_unfolding_dim': row_dict.get('cuboid_unfolding_dim'),
                        'cuboid_unfolding_n0': row_dict.get('cuboid_unfolding_n0'),
                        'cylinder_unfolding_dim': row_dict.get('cylinder_unfolding_dim'),
                        'cylinder_unfolding_n0': row_dict.get('cylinder_unfolding_n0'),
                        # === 新增数据 ===
                        'slice_profile_dim': slice_profile_dim,
                        'slice_profile_n0': slice_profile_n0,
                        
                        'repeat_index': row_dict.get('repeat_index'), 'success': bool(row_dict.get('success', False)),
                        'timestamp': row_dict.get('timestamp'), 'process_id': row_dict.get('process_id'),
                        'task_info': json.loads(row_dict.get('task_info')) if row_dict.get('task_info') else None
                    })
                
                return {'metadata': metadata, 'results': results}
            finally:
                conn.close()
        except Exception as e:
            print(f"加载备份数据失败: {e}")
            return None

class TempFileManager:
    """
    临时文件管理器，用于处理多进程计算时进程间的文件交换。
    """
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="fractal_calc_"))
        self.result_files = []
        print(f"创建临时目录: {self.temp_dir}")

    def get_result_file_path(self, task_id, repeat_index):
        """获取结果文件路径"""
        filename = f"result_{task_id}_{repeat_index}.json"
        filepath = self.temp_dir / filename
        self.result_files.append(filepath)
        return str(filepath)

    def cleanup(self):
        """清理所有临时文件和目录"""
        cleaned_files = 0
        for file_path in self.result_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleaned_files += 1
                except Exception as e:
                    print(f"清理文件失败 {file_path}: {e}")
        
        if self.temp_dir.exists():
            try:
                self.temp_dir.rmdir()
                print(f"清理临时目录: {self.temp_dir}, 清理了{cleaned_files}个文件")
            except Exception as e:
                print(f"清理临时目录失败 {self.temp_dir}: {e}")

# --- END OF FILE data_manager.py ---