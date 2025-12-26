好的，我已经仔细阅读并分析了您提供的所有代码。这是一个功能强大且结构完善的软件项目。以下是为您生成的最新 `README.md` 文档和详细的 `API` 文档。

---

# README.md

## Advanced 3D Fractal Fracture Network Generator (高级3D分形裂缝网络生成器)

**版本: 2.0**

这是一个高级的科学计算与可视化软件，旨在基于分形维数理论生成、分析和可视化三维空间中的复杂椭圆裂缝网络。该软件适用于地质力学、岩石工程、石油和天然气勘探等需要对地下裂缝系统进行建模和分析的领域。

 <!-- 建议您在这里替换为软件的实际截图 -->

### ✨ 核心功能

*   **分形驱动的生成**: 基于用户定义的分形维数 (D) 和分形初值 (N0) 生成符合幂律分布的三维裂缝网络。
*   **形态参数控制**: 精确控制裂缝的形态，包括长短轴比例、倾角和方位角，支持固定值或随机分布。
*   **多维度分形分析**:
    *   **3D分形维数验证**: 使用盒计数法（Box-Counting Method）计算生成网络的三维分形维数，并与理论输入值进行对比验证。
    *   **2D分形维数分析**: 对模型的任意正交截面（XY, YZ, XZ）进行切片，分析二维迹线的分形特征。
*   **批量模拟与并行计算**:
    *   内置“2D-3D关系研究”模块，支持对多组参数进行批量化、重复性模拟。
    *   利用多进程并行计算，极大地缩短了大规模模拟所需的时间。
*   **数据持久化与恢复**:
    *   所有批量计算结果都会被实时备份到 **SQLite** 数据库中，防止因意外中断造成数据丢失。
    *   支持从备份文件加载历史结果，并可以从上次中断的位置继续进行计算。
*   **交互式钻孔模拟**:
    *   在生成的三维裂缝网络中，模拟垂直钻孔过程。
    *   交互式地分析钻孔与哪些裂缝相交，并以3D视图和表格形式展示详细的相交信息。
*   **高级可视化**:
    *   **迭代过程可视化**: 在不同的选项卡中清晰地展示分形迭代的每一个步骤。
    *   **统计分析图表**: 自动生成并展示裂缝倾角、方位角的分布直方图。
    *   **截面捕获**: 可通过滑动条实时查看任意位置的截面迹线图，并能自动计算该截面的分形维数。
*   **多格式模型导出**:
    *   支持将生成的三维裂缝网络模型导出为多种标准格式，包括 **STL, OBJ, PLY**，方便在其他三维软件（如ParaView, Blender）中进行后续分析。
    *   支持将裂缝的详细参数（中心、法向量、长短轴等）导出为 **TXT** 数据文件。
*   **国际化支持**: 提供中文和英文两种语言界面，方便不同用户使用。

### ⚙️ 技术架构

软件采用模块化设计，各部分职责分明：

*   **主程序 (`FD-Advanced_3D.py`)**: 基于 **PyQt5** 的图形用户界面，负责用户交互和整体流程控制。
*   **裂缝模型 (`fracture_model.py`)**: 定义了 `EllipticalFracture` 核心数据结构和 `FractalBasedFractureGenerator` 裂缝生成算法。
*   **分形分析 (`fractal_analysis.py`)**: 包含了 `FractalDimension3DCalculator` 和 `FractalDimension2DCalculator`，实现了核心的盒计数算法。
*   **钻孔分析 (`drilling_analyzer.py`)**: 实现了钻孔与裂缝网络相交的几何分析逻辑。
*   **数据管理 (`data_manager.py`)**: 通过 `SQLiteBackupManager` 提供了强大的多进程数据备份与恢复能力。
*   **模型导出 (`exporters.py`)**: 封装了将裂缝数据转换为不同文件格式的逻辑。
*   **随机数管理 (`random_manager.py`)**: 确保了在复杂模拟中随机数的可复现性。
*   **国际化 (`translation_manager.py`)**: 管理UI语言的切换。

### 🚀 安装与运行

#### 依赖库

请确保您已安装以下Python库。推荐使用 `pip`进行安装：

```bash
pip install numpy
pip install matplotlib
pip install PyQt5
pip install pandas
pip install open3d
```

#### 运行程序

1.  将所有代码文件放在同一个目录下。
2.  确保 `translations` 文件夹及内部的 `en.qm` 文件与主程序在同一级目录结构下。
3.  执行主程序文件：

```bash
python FD-Advanced_3D.py
```

### 📖 使用指南

1.  **参数设置**: 在程序左侧的“分形维数参数”和“裂缝形态参数”面板中，设置您期望的模拟参数。
2.  **生成裂缝**: 点击 **“生成分形裂缝”** 按钮。程序将在主显示区的不同选项卡中展示迭代生成的裂缝网络。
3.  **分析与验证**:
    *   生成后，点击 **“计算分形维数”** 按钮，软件将验证生成模型的3D分形维数，并在“统计信息”区域显示详细的对比结果。
    *   切换到 **“统计分析”** 选项卡，查看裂缝的倾角、方位角分布以及3D分形维数拟合曲线。
    *   切换到 **“...个投影面轨迹分析”** 选项卡，查看不同方向切片的平均分形特征。
4.  **批量研究**:
    *   点击 **“2D-3D关系研究”** 按钮，打开批量模拟面板。
    *   设置参数范围（支持单个值、空格分隔的多个值或 `(start, step, end)` 格式），选择是否启用并行计算。
    *   点击 **“开始计算”**。计算结果将实时显示在表格中，并自动备份。
    *   若计算中断，可通过 **“加载备份文件”** 按钮恢复进度。
5.  **钻孔模拟**:
    *   生成裂缝网络后，点击 **“钻孔模拟研究”** 按钮，打开钻孔分析面板。
    *   点击 **“生成随机点”** 在模型顶面创建一个钻孔位置。
    *   点击 **“开始钻孔分析”**，软件将模拟钻孔过程，并在3D视图中标红穿过的裂缝，同时在下方表格列出详细信息。
6.  **模型导出**:
    *   在左侧“模型导出”面板中选择所需格式。
    *   点击 **“导出模型”** 按钮，将当前的三维裂缝网络保存到本地文件。

---

# API 文档

本文档提供了对“高级3D分形裂缝网络生成器”项目中核心模块和类的API说明。

### 1. `fracture_model.py` - 核心模型与生成器

#### `class EllipticalFracture`

表示一个独立的三维椭圆裂缝。

*   `__init__(self, center, normal, semi_major_axis, ...)`
    *   **描述**: 初始化一个裂缝对象。
    *   **参数**:
        *   `center` (array): 裂缝中心的三维坐标 `[x, y, z]`。
        *   `semi_major_axis` (float): 椭圆长半轴。
        *   `semi_minor_axis` (float): 椭圆短半轴。
        *   `inclination` (float): 倾角 (弧度)。
        *   `azimuth` (float): 方位角 (弧度)。
        *   `iteration_level` (int): 该裂缝所属的生成迭代层级。
        *   `fracture_id` (int): 唯一的裂缝ID。
*   `clip_to_cube(self, cube_size)`
    *   **描述**: 将裂缝的多边形顶点裁切到由 `cube_size` 定义的立方体边界内。同时会计算并存储裂缝与立方体边界的交线（迹线），用于后续的2D分析。
    *   **返回**: 裁切后的顶点数组 (`numpy.ndarray`)。
*   `get_area_in_box(self, box_min, box_max)`
    *   **描述**: 高效地计算该裂缝在指定边界盒（box）内的面积。内部实现了缓存和快速相交检查以优化性能。
    *   **返回**: (float) 面积值。

#### `class FractalBasedFractureGenerator`

实现了基于分形理论的裂缝网络生成算法。

*   `generate_fractures(self, cube_size, fractal_dimension, n0, ...)`
    *   **描述**: 核心生成函数。根据输入的分形参数和形态参数，通过多轮迭代生成一个完整的裂缝网络。
    *   **参数**:
        *   `cube_size` (float): 模拟立方体的边长。
        *   `fractal_dimension` (float): 理论分形维数 D。
        *   `n0` (float): 理论分形初值 N0。
        *   `max_iterations` (int): 生成迭代的总次数。
        *   `rng_manager` (RandomStateManager): 随机数管理器实例。
        *   `**kwargs`: 其他形态参数。
    *   **返回**: `(fractures_by_level, structured_log)` 元组，包含按层级组织的裂缝字典和详细的结构化生成日志。
*   `get_fractures_up_to_level(self, max_level)`
    *   **描述**: 从已生成的结果中，获取从第0层到 `max_level` 层的全部裂缝列表。
    *   **返回**: (list) `EllipticalFracture` 对象列表。

### 2. `fractal_analysis.py` - 分形维数计算

#### `class FractalDimension3DCalculator`

使用盒计数法计算三维裂缝网络的分形维数。

*   `calculate_fractal_dimension_3d(self, fractures, cube_size)`
    *   **描述**: 对给定的裂缝列表执行3D盒计数法分析。
    *   **返回**: `(dimension, fit_data, levels)` 元组。
        *   `dimension` (float): 计算出的分形维数。
        *   `fit_data` (dict): 拟合数据，包括对数坐标点、拟合系数和R²值。
        *   `levels` (list): 每个计算层级的详细数据（盒子大小、有效计数等）。

#### `class FractalDimension2DCalculator`

使用盒计数法计算二维裂缝迹线的分形维数。

*   `calculate_fractal_dimension(self, line_segments, cube_size, ...)`
    *   **描述**: 对给定的二维线段集合（裂缝迹线）执行2D盒计数法分析。
    *   **返回**: 与 `FractalDimension3DCalculator` 结构相同的元组。

#### `class FractalAnalysisUtils`

提供可复用的静态分析方法。

*   `calculate_average_2d_dimension(fractures, cube_size, ...)` (static)
    *   **描述**: 这是一个重构的辅助方法，通过对模型进行全方位的密集切片，计算所有截面迹线的平均2D分形维数和N0值。
    *   **返回**: `(avg_2d_dim, avg_2d_n0)` 元组。

### 3. `data_manager.py` - 数据持久化

#### `class SQLiteBackupManager`

一个为多进程环境设计的、健壮的SQLite备份管理器。

*   `__init__(self, backup_file_path=None, calculation_params=None)`
    *   **描述**: 初始化或连接到一个SQLite数据库。如果未提供路径，则会自动在 `FD-Advanced_3D_temp_data` 目录下创建一个带时间戳的数据库文件。它会自动创建所需的 `metadata` 和 `results` 表。
*   `append_result(self, result_data, ...)`
    *   **描述**: 将单次模拟的结果（一个字典）追加到数据库中。此方法是**进程安全**的，内部使用了 `PRAGMA journal_mode=WAL` 和重试机制来处理数据库锁定问题。
*   `load_backup_data(self)`
    *   **描述**: 从数据库中加载所有元数据和结果数据。
    *   **返回**: 一个包含 `'metadata'` 和 `'results'` 键的字典。

### 4. `drilling_analyzer.py` - 钻孔模拟分析

#### `class DrillingAnalyzer`

负责模拟垂直钻孔与三维裂缝网络的几何相交分析。

*   `generate_random_point(self, cube_size)`
    *   **描述**: 在立方体的顶面（XY平面）上生成一个随机的钻孔起始点。
*   `analyze_drilling_probability(self, fractures, cube_size, random_point)`
    *   **描述**: 核心分析函数。计算从 `random_point` 开始的垂直钻孔会与裂缝列表中的哪些裂缝相交。
    *   **返回**: (dict) 一个包含详细分析结果的字典，包括：
        *   `'random_point'`: 钻孔起点。
        *   `'intersected_fractures'`: 一个列表，每个元素都是一个包含相交裂缝对象、交点坐标等信息的字典。
        *   `'total_intersections'`: 相交总数。

### 5. `exporters.py` - 模型与数据导出

#### `class FractureExporter`

提供将裂缝网络导出为不同文件格式的功能。

*   `export_to_stl(self, fractures, cube_size, filename)`
*   `export_to_obj(self, fractures, cube_size, filename)`
*   `export_to_ply(self, fractures, cube_size, filename)`
    *   **描述**: 将裂缝网络（包含立方体线框）导出为标准的三维模型文件。
*   `export_fracture_data(self, fractures, cube_size, filename)`
    *   **描述**: 将每个裂缝的详细参数（中心、法向量、长短轴、面积、顶点坐标等）导出为人类可读的TXT文本文件。

### 6. `random_manager.py` - 随机数管理

#### `class RandomStateManager`

统一的随机状态管理器，确保模拟的可复现性。

*   `__init__(self, master_seed)`
    *   **描述**: 使用一个主种子来初始化多个独立的随机数生成器。
    *   **属性**:
        *   `self.fracture_rng`: 用于裂缝生成过程。
        *   `self.placement_rng`: 用于放置研究盒子（当前代码中未使用，但为未来扩展保留）。
        *   `self.drilling_rng`: 用于生成钻孔的随机位置。

### 7. `FD-Advanced_3D.py` - GUI与主逻辑

#### `class FractalFractureGeneratorGUI`

应用程序的主窗口类，管理所有UI组件和业务逻辑。

*   **核心线程**:
    *   `FractureGenerationThread`: 在后台线程中执行裂缝生成，防止UI冻结。
    *   `FractalDimensionCalculationThread`: 在后台线程中执行3D分形维数计算。
    *   `Study2D3DParallelCalculationThread`: **多进程控制器**。负责生成任务队列，创建并管理一个 `multiprocessing.Pool` 来并行执行批量模拟任务。通过共享的 `multiprocessing.Event` 实现优雅停止。
*   **核心函数 (批量计算)**:
    *   `start_2d3d_calculation()`: 解析UI参数，创建并启动 `Study2D3D...Thread`。
    *   `stop_2d3d_calculation()`: 请求停止计算，并等待所有子进程结束后同步最终数据。
    *   `load_backup_file()`: 加载SQLite备份文件，分析已完成的任务，并询问用户是否继续计算。
    *   `on_study_result_received_and_update_ui(self, result)`: **统一的结果处理槽函数**。无论是单线程还是多进程，所有完成的单个结果最终都会通过信号发送到此函数，由主GUI线程安全地更新UI（表格、进度条等）。