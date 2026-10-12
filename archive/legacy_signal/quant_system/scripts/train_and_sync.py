"""
Windows端训练+同步脚本
训练完成后自动推送模型到SMB共享目录
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import WATCHLIST
from models.ml_trainer import MLTrainer
from sync.smb_sync import SMBSyncManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainAndSync")


def train_and_sync(
    smb_remote: str = "//192.168.1.100/quant_sync",
    symbols: list = None,
    tune: bool = False,
    push_db: bool = True,
):
    """
    完整流程：训练模型 → 评估 → 推送SMB
    
    Args:
        smb_remote: SMB共享路径，如 "//192.168.1.100/quant_sync"
        symbols: 训练股票池，默认WATCHLIST前50只
        tune: 是否启用Optuna超参调优
        push_db: 是否同时推送数据库
    """
    symbols = symbols or WATCHLIST[:50]
    
    logger.info("=" * 60)
    logger.info("Windows训练工厂 | %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # 1. 训练模型
    logger.info("[1/4] 开始训练多视野回归模型...")
    trainer = MLTrainer()
    train_result = trainer.train_model(symbols=symbols, tune=tune)
    
    # 2. 评估模型
    logger.info("[2/4] 评估模型衰减...")
    eval_result = trainer.evaluate_model(symbols=symbols, n_days=30)
    
    if eval_result.get("overall_needs_retrain"):
        logger.warning("模型衰减严重，建议检查后再同步")
        # 可以选择在这里中断，不推送差模型
        # return
    
    # 3. 本地备份
    logger.info("[3/4] 备份旧模型...")
    _backup_models()
    
    # 4. 推送到SMB
    logger.info("[4/4] 推送到SMB共享...")
    sync = SMBSyncManager(
        local_root=str(Path(__file__).resolve().parent.parent),
        remote_root=smb_remote,
        remote_type="smb_share",
    )
    
    result = sync.sync_all_push()
    logger.info("推送完成:")
    logger.info("  模型文件: %d个", len(result["models"]))
    logger.info("  数据库: %s", "已推送" if result["database"] else "未变化/跳过")
    logger.info("  时间: %s", result["timestamp"])
    
    logger.info("=" * 60)
    logger.info("训练+同步全部完成")
    logger.info("=" * 60)
    
    return {"train": train_result, "eval": eval_result, "sync": result}


def _backup_models():
    """备份当前模型文件"""
    from config.settings import MODELS_DIR
    import shutil
    
    backup_dir = MODELS_DIR / "backup"
    backup_dir.mkdir(exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = backup_dir / ts
    backup_subdir.mkdir(exist_ok=True)
    
    for f in MODELS_DIR.glob("*.txt"):
        if f.is_file():
            shutil.copy2(f, backup_subdir / f.name)
    for f in MODELS_DIR.glob("*.json"):
        if f.is_file():
            shutil.copy2(f, backup_subdir / f.name)
    
    logger.info("模型已备份到: %s", backup_subdir)


def schedule_train_and_sync():
    """
    定时训练+同步（建议每周运行一次）
    可以用Windows任务计划程序或cron调用此脚本
    """
    import schedule
    import time
    
    # 每周日凌晨2点训练+同步
    schedule.every().sunday.at("02:00").do(train_and_sync)
    
    logger.info("定时训练任务已设置: 每周日 02:00")
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Windows训练+SMB同步")
    parser.add_argument("--smb", default="//192.168.1.100/quant_sync", help="SMB共享路径")
    parser.add_argument("--tune", action="store_true", help="启用超参调优")
    parser.add_argument("--no-db", action="store_true", help="不推送数据库")
    parser.add_argument("--schedule", action="store_true", help="进入定时模式")
    
    args = parser.parse_args()
    
    if args.schedule:
        schedule_train_and_sync()
    else:
        train_and_sync(
            smb_remote=args.smb,
            tune=args.tune,
            push_db=not args.no_db,
        )
