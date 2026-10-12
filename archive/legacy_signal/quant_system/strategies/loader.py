"""
A股量化信号系统 - 策略动态加载器

支持从以下位置加载策略:
1. 内置策略 (strategies.built_in)
2. 自定义策略文件 (strategies/custom/*.py)
3. 外部策略文件 (任意路径的 .py 文件)

使用方式:
    from strategies.loader import StrategyLoader

    loader = StrategyLoader()

    # 加载内置策略
    loader.load_built_in()

    # 加载自定义策略目录
    loader.load_from_directory("strategies/custom")

    # 加载单个策略文件
    loader.load_from_file("my_strategy.py")

    # 获取注册中心
    registry = loader.get_registry()
    print(f"已加载 {len(registry)} 个策略")
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

from strategies.base import BaseStrategy
from strategies.built_in import register_all_built_in
from strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


class StrategyLoader:
    """
    策略动态加载器
    自动发现和注册策略类
    """

    def __init__(self):
        self.registry = StrategyRegistry()
        self._loaded_files: set = set()

    def get_registry(self) -> StrategyRegistry:
        """获取策略注册中心"""
        return self.registry

    # ============================================================
    # 内置策略
    # ============================================================

    def load_built_in(self, **kwargs) -> "StrategyLoader":
        """
        加载所有内置策略

        Args:
            **kwargs: 传递给内置策略的参数，如 ma_fast=10
        """
        register_all_built_in(self.registry, **kwargs)
        logger.info("内置策略加载完成，共 %d 个", len(self.registry))
        return self

    # ============================================================
    # 从目录加载
    # ============================================================

    def load_from_directory(self, dir_path: str) -> "StrategyLoader":
        """
        从目录加载所有策略文件

        Args:
            dir_path: 策略文件所在目录路径
        """
        path = Path(dir_path)
        if not path.exists():
            logger.warning("策略目录不存在: %s", dir_path)
            return self

        py_files = sorted(path.glob("*_strategy.py"))
        if not py_files:
            py_files = sorted(path.glob("*.py"))

        logger.info("发现 %d 个策略文件在 %s", len(py_files), dir_path)

        for py_file in py_files:
            if py_file.name.startswith("__"):
                continue
            self.load_from_file(str(py_file))

        return self

    # ============================================================
    # 从文件加载
    # ============================================================

    def load_from_file(self, file_path: str) -> bool:
        """
        从单个 .py 文件加载策略

        Args:
            file_path: 策略文件路径

        Returns:
            是否成功加载至少一个策略
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("策略文件不存在: %s", file_path)
            return False

        abs_path = path.resolve()
        if str(abs_path) in self._loaded_files:
            logger.debug("策略文件已加载过，跳过: %s", file_path)
            return True

        try:
            # 动态导入模块
            module_name = f"dynamic_strategy_{abs_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self._loaded_files.add(str(abs_path))

            # 查找模块中所有 BaseStrategy 子类
            loaded_count = 0
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseStrategy)
                    and attr is not BaseStrategy
                ):
                    try:
                        # 实例化策略（使用默认参数）
                        strategy = attr()
                        self.registry.register(strategy)
                        loaded_count += 1
                        logger.info(
                            "从 %s 加载策略: %s (%s)",
                            path.name,
                            strategy.name,
                            strategy.display_name,
                        )
                    except Exception as e:
                        logger.error(
                            "策略实例化失败 %s.%s: %s",
                            module_name,
                            attr_name,
                            e,
                        )

            if loaded_count == 0:
                logger.warning("文件 %s 中未发现策略类", path.name)

            return loaded_count > 0

        except Exception as e:
            logger.error("加载策略文件失败 %s: %s", file_path, e)
            return False

    # ============================================================
    # 便捷方法
    # ============================================================

    def load_all(self, custom_dir: Optional[str] = None, **built_in_kwargs) -> "StrategyLoader":
        """
        一键加载所有策略（内置 + 自定义）

        Args:
            custom_dir: 自定义策略目录路径
            **built_in_kwargs: 内置策略参数
        """
        self.load_built_in(**built_in_kwargs)
        if custom_dir:
            self.load_from_directory(custom_dir)
        logger.info("策略加载完成，共 %d 个", len(self.registry))
        return self

    def list_loaded_files(self) -> list:
        """列出已加载的策略文件"""
        return sorted(self._loaded_files)

    def reload(self, file_path: str) -> bool:
        """
        重新加载某个策略文件（开发时热更新）
        """
        abs_path = str(Path(file_path).resolve())
        if abs_path in self._loaded_files:
            self._loaded_files.discard(abs_path)
        return self.load_from_file(file_path)
