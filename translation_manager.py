# translation_manager.py (最终修正版)
from PyQt5.QtCore import QObject, QTranslator, QCoreApplication, pyqtSignal
import os

class TranslationManager(QObject):
    # 定义一个信号，当语言改变时发射
    language_changed = pyqtSignal()

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.translator = QTranslator(self.app)
        self.current_lang = 'zh' # 默认中文

    def load_translation(self, lang_code):
        """
        加载并安装指定语言的翻译文件
        lang_code: 'en' or 'zh'
        """
        # 先移除旧的翻译器
        self.app.removeTranslator(self.translator)
        
        if lang_code == 'en':
            # --- 核心修改在这里 ---
            # 1. 获取当前脚本 (translation_manager.py) 所在的目录
            #    __file__ 是一个指向当前文件路径的 Python 内置变量
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 2. 基于此目录构造翻译文件的绝对路径
            #    这样可以确保无论从哪里运行程序，路径都是正确的
            path = os.path.join(base_dir, "translations", "en.qm")
            # --- 修改结束 ---
            
            if self.translator.load(path):
                self.app.installTranslator(self.translator)
                self.current_lang = 'en'
                print(f"Successfully loaded English translation from: {path}")
            else:
                print(f"Failed to load English translation from: {path}")
        else:
            # 切换回中文时，只需移除翻译器即可恢复默认文本
            self.current_lang = 'zh'
            print("Switched back to Chinese (default).")

        # 发射信号，通知UI更新文本
        self.language_changed.emit()