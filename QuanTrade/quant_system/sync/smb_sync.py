"""
SMB 同步模块
支持 Windows ↔ 树莓派 模型/数据/预测图自动同步
"""

import hashlib
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SMBSyncManager:
    """
    SMB 同步管理器
    
    架构:
        Windows (训练端)                    树莓派 (服务端)
        ├─ models/  ←── SMB共享 ──→  挂载为 /mnt/quant_models
        ├─ data/quant.db  ←── SMB ──→  /mnt/quant_data/quant.db
        └─ plots/  ←── SMB ──→  /mnt/quant_plots
    
    使用方式:
        # Windows端（推送）
        sync = SMBSyncManager(local_root=".", remote_type="smb_share")
        sync.push_models()
        sync.push_database()
        
        # 树莓派端（拉取/检测）
        sync = SMBSyncManager(local_root=".", remote_type="smb_mount")
        sync.pull_models_if_changed()
    """

    # 需要同步的文件模式
    MODEL_PATTERNS = [
        "lgb_regressor_*.txt",
        "lgb_volatility.txt",
        "model_config.json",
        "model_config_*.json",
    ]
    
    DB_FILES = ["data/quant.db"]
    PLOT_PATTERNS = ["plots/*.png"]

    def __init__(
        self,
        local_root: str = ".",
        remote_root: Optional[str] = None,
        remote_type: str = "smb_mount",  # "smb_mount" | "smb_share" | "local_copy"
    ):
        """
        Args:
            local_root: 本地项目根目录
            remote_root: 远程目录路径
                - smb_mount模式: 树莓派上SMB挂载点，如 "/mnt/quant_sync"
                - smb_share模式: Windows共享路径，如 "//192.168.1.100/quant_sync"
                - local_copy模式: 本地另一个目录，用于测试
            remote_type: 同步模式
        """
        self.local_root = Path(local_root).resolve()
        self.remote_root = Path(remote_root) if remote_root else None
        self.remote_type = remote_type
        
        # 状态追踪文件（记录上次同步的哈希）
        self.state_file = self.local_root / ".sync_state.json"
        self._state = self._load_state()

    def _load_state(self) -> Dict:
        """加载同步状态"""
        import json
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_state(self):
        """保存同步状态"""
        import json
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)

    def _file_hash(self, path: Path) -> str:
        """计算文件MD5哈希"""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _find_files(self, patterns: List[str], root: Path) -> Dict[str, Path]:
        """根据模式查找文件，返回 {relative_path: absolute_path}"""
        files = {}
        for pattern in patterns:
            for p in root.rglob(pattern):
                if p.is_file():
                    rel = str(p.relative_to(root))
                    files[rel] = p
        return files

    # ============================================================
    # 推送（Windows端 → 远程）
    # ============================================================

    def push_models(self) -> List[str]:
        """推送模型文件到远程"""
        if self.remote_root is None:
            logger.warning("remote_root未设置，跳过推送")
            return []

        synced = []
        local_models = self._find_files(self.MODEL_PATTERNS, self.local_root / "models")
        
        for rel_path, local_path in local_models.items():
            remote_path = self.remote_root / "models" / rel_path
            remote_path.parent.mkdir(parents=True, exist_ok=True)
            
            current_hash = self._file_hash(local_path)
            last_hash = self._state.get(f"models/{rel_path}")
            
            if current_hash != last_hash:
                shutil.copy2(local_path, remote_path)
                self._state[f"models/{rel_path}"] = current_hash
                synced.append(rel_path)
                logger.info("模型已同步: %s", rel_path)
            else:
                logger.debug("模型未变化，跳过: %s", rel_path)
        
        if synced:
            self._save_state()
            # 写入同步时间戳标记
            marker = self.remote_root / "models" / ".last_sync"
            marker.write_text(datetime.now().isoformat(), encoding="utf-8")
        
        return synced

    def push_database(self) -> bool:
        """推送数据库文件到远程（可选，如果树莓派需要最新数据）"""
        if self.remote_root is None:
            return False

        db_path = self.local_root / "data" / "quant.db"
        if not db_path.exists():
            logger.warning("数据库不存在: %s", db_path)
            return False

        remote_db = self.remote_root / "data" / "quant.db"
        remote_db.parent.mkdir(parents=True, exist_ok=True)
        
        current_hash = self._file_hash(db_path)
        last_hash = self._state.get("data/quant.db")
        
        if current_hash != last_hash:
            shutil.copy2(db_path, remote_db)
            self._state["data/quant.db"] = current_hash
            self._save_state()
            logger.info("数据库已同步 | 大小: %.1f MB", db_path.stat().st_size / 1024 / 1024)
            return True
        else:
            logger.debug("数据库未变化，跳过")
            return False

    def push_plots(self) -> List[str]:
        """推送预测图到远程"""
        if self.remote_root is None:
            return []

        synced = []
        local_plots = self._find_files(self.PLOT_PATTERNS, self.local_root)
        
        for rel_path, local_path in local_plots.items():
            remote_path = self.remote_root / rel_path
            remote_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 预测图总是推送（因为每次生成都是新的）
            shutil.copy2(local_path, remote_path)
            synced.append(rel_path)
            logger.info("预测图已同步: %s", rel_path)
        
        return synced

    # ============================================================
    # 拉取（树莓派端 ← 远程）
    # ============================================================

    def pull_models_if_changed(self) -> List[str]:
        """
        检测远程模型是否有更新，有则拉取到本地
        树莓派端使用此方法
        """
        if self.remote_root is None:
            logger.warning("remote_root未设置，跳过拉取")
            return []

        synced = []
        remote_models_dir = self.remote_root / "models"
        local_models_dir = self.local_root / "models"
        
        if not remote_models_dir.exists():
            logger.warning("远程模型目录不存在: %s", remote_models_dir)
            return []

        # 检查同步标记
        marker = remote_models_dir / ".last_sync"
        if marker.exists():
            remote_sync_time = marker.read_text(encoding="utf-8").strip()
            local_sync_time = self._state.get("last_pull_time", "")
            
            if remote_sync_time == local_sync_time:
                logger.info("远程模型未更新，跳过拉取")
                return []

        # 拉取所有模型文件
        for pattern in self.MODEL_PATTERNS:
            for remote_path in remote_models_dir.rglob(pattern):
                if not remote_path.is_file():
                    continue
                
                rel_path = str(remote_path.relative_to(remote_models_dir))
                local_path = local_models_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                remote_hash = self._file_hash(remote_path)
                local_hash = self._file_hash(local_path) if local_path.exists() else ""
                
                if remote_hash != local_hash:
                    shutil.copy2(remote_path, local_path)
                    synced.append(rel_path)
                    logger.info("模型已拉取: %s", rel_path)

        if synced:
            if marker.exists():
                self._state["last_pull_time"] = marker.read_text(encoding="utf-8").strip()
            self._save_state()
            logger.info("模型拉取完成: %d个文件更新", len(synced))
        else:
            logger.info("所有模型已是最新")
        
        return synced

    def pull_database_if_changed(self) -> bool:
        """检测远程数据库是否有更新，有则拉取"""
        if self.remote_root is None:
            return False

        remote_db = self.remote_root / "data" / "quant.db"
        local_db = self.local_root / "data" / "quant.db"
        
        if not remote_db.exists():
            return False

        remote_hash = self._file_hash(remote_db)
        local_hash = self._file_hash(local_db) if local_db.exists() else ""
        
        if remote_hash != local_hash:
            local_db.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_db, local_db)
            logger.info("数据库已拉取 | 大小: %.1f MB", remote_db.stat().st_size / 1024 / 1024)
            return True
        
        logger.debug("数据库已是最新")
        return False

    # ============================================================
    # 便捷方法
    # ============================================================

    def sync_all_push(self) -> Dict:
        """Windows端一键推送所有"""
        return {
            "models": self.push_models(),
            "database": self.push_database(),
            "timestamp": datetime.now().isoformat(),
        }

    def sync_all_pull(self) -> Dict:
        """树莓派端一键拉取所有"""
        return {
            "models": self.pull_models_if_changed(),
            "database": self.pull_database_if_changed(),
            "timestamp": datetime.now().isoformat(),
        }

    def wait_for_remote_update(self, timeout: int = 300, interval: int = 10) -> bool:
        """
        树莓派端：阻塞等待远程模型更新
        用于训练完成后立即同步的场景
        """
        logger.info("等待远程模型更新 (超时%d秒)...", timeout)
        start = time.time()
        
        while time.time() - start < timeout:
            result = self.sync_all_pull()
            if result["models"]:
                logger.info("检测到模型更新: %s", result["models"])
                return True
            time.sleep(interval)
        
        logger.warning("等待超时，未检测到模型更新")
        return False


# ============================================================
# 命令行接口
# ============================================================

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="SMB同步工具")
    parser.add_argument("--mode", choices=["push", "pull", "watch"], required=True)
    parser.add_argument("--local", default=".", help="本地根目录")
    parser.add_argument("--remote", required=True, help="远程目录路径")
    parser.add_argument("--type", default="smb_mount", choices=["smb_mount", "smb_share", "local_copy"])
    parser.add_argument("--watch-interval", type=int, default=60, help="监控间隔秒数")
    
    args = parser.parse_args()

    sync = SMBSyncManager(
        local_root=args.local,
        remote_root=args.remote,
        remote_type=args.type,
    )

    if args.mode == "push":
        result = sync.sync_all_push()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.mode == "pull":
        result = sync.sync_all_pull()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.mode == "watch":
        # 树莓派持续监控模式
        print(f"进入监控模式，每{args.watch_interval}秒检测一次...")
        while True:
            result = sync.sync_all_pull()
            if result["models"]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 模型更新: {result['models']}")
            time.sleep(args.watch_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
