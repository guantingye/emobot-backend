# migration_script.py
"""
資料庫遷移腳本 - 新增 allowed_pids 和 chat_sessions 表
執行方式：python migration_script.py
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 添加專案根目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 載入環境變數
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ★ 修復：更改導入順序，避免循環引用
from app.db.session import DATABASE_URL, engine
from app.db.base import Base

# ★ 修復：分別導入每個模型，確保正確的初始化順序
from app.models.user import User
from app.models.assessment import Assessment
from app.models.recommendation import Recommendation
from app.models.chat import ChatMessage
from app.models.mood import MoodRecord

# ★ 最後導入新模型
from app.models.allowed_pid import AllowedPid
from app.models.chat_session import ChatSession

def run_migration():
    """執行資料庫遷移"""
    print("🔄 開始資料庫遷移...")
    
    try:
        # 建立新的表格
        print("📝 建立新表格...")
        Base.metadata.create_all(bind=engine)
        print("✅ 表格建立完成")
        
        # 建立資料庫會話
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        try:
            # 新增預設的允許 PID（你可以根據需要修改）
            print("👥 新增預設允許的 PID...")
            default_pids = [
                {"pid": "123A", "description": "測試用戶 A"},
                {"pid": "456B", "description": "測試用戶 B"},
                {"pid": "789C", "description": "測試用戶 C"},
                # 根據你的需要添加更多 PID
            ]
            
            for pid_data in default_pids:
                existing = db.query(AllowedPid).filter(AllowedPid.pid == pid_data["pid"]).first()
                if not existing:
                    allowed_pid = AllowedPid(
                        pid=pid_data["pid"],
                        description=pid_data["description"],
                        is_active=True
                    )
                    db.add(allowed_pid)
                    print(f"  ✅ 新增 PID: {pid_data['pid']}")
                else:
                    print(f"  ⚠️  PID 已存在: {pid_data['pid']}")
            
            # 提交變更
            db.commit()
            print("💾 預設資料新增完成")
            
            # 驗證表格是否正確建立
            print("🔍 驗證表格結構...")
            with engine.connect() as conn:
                # 檢查 allowed_pids 表
                result = conn.execute(text("SELECT COUNT(*) FROM allowed_pids"))
                allowed_count = result.scalar()
                print(f"  📊 allowed_pids 表: {allowed_count} 筆記錄")
                
                # 檢查 chat_sessions 表
                result = conn.execute(text("SELECT COUNT(*) FROM chat_sessions"))
                session_count = result.scalar()
                print(f"  📊 chat_sessions 表: {session_count} 筆記錄")
            
            print("🎉 資料庫遷移完成！")
            
        except Exception as e:
            print(f"❌ 資料操作失敗: {e}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ 遷移失敗: {e}")
        raise

def add_sample_pids():
    """新增範例 PID（可選執行）"""
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        print("📝 新增範例 PID...")
        sample_pids = [
            "001A", "002B", "003C", "004D", "005E",
            "100X", "200Y", "300Z", "400W", "500V"
        ]
        
        for pid in sample_pids:
            existing = db.query(AllowedPid).filter(AllowedPid.pid == pid).first()
            if not existing:
                allowed_pid = AllowedPid(
                    pid=pid,
                    description=f"範例用戶 {pid}",
                    is_active=True
                )
                db.add(allowed_pid)
                print(f"  ✅ 新增範例 PID: {pid}")
        
        db.commit()
        print("✅ 範例 PID 新增完成")
        
    except Exception as e:
        print(f"❌ 新增範例 PID 失敗: {e}")
        db.rollback()
    finally:
        db.close()

def test_database_connection():
    """測試資料庫連接"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ 資料庫連接測試成功")
            return True
    except Exception as e:
        print(f"❌ 資料庫連接測試失敗: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Emobot+ 資料庫遷移工具")
    print("=" * 50)
    
    # 顯示當前資料庫 URL（隱藏敏感資訊）
    masked_url = DATABASE_URL[:20] + "***" + DATABASE_URL[-10:] if len(DATABASE_URL) > 30 else "本地資料庫"
    print(f"📍 資料庫: {masked_url}")
    
    # 測試資料庫連接
    if not test_database_connection():
        print("❌ 資料庫連接失敗，請檢查設定")
        sys.exit(1)
    
    # 執行主要遷移
    try:
        run_migration()
    except Exception as e:
        print(f"\n❌ 遷移過程發生錯誤: {e}")
        sys.exit(1)
    
    # 詢問是否新增範例 PID
    try:
        add_samples = input("\n❓ 是否新增範例 PID (10個)？(y/N): ").lower().strip()
        if add_samples in ['y', 'yes']:
            add_sample_pids()
    except KeyboardInterrupt:
        print("\n⚠️ 用戶取消操作")
    
    print("\n🎯 遷移完成！現在你可以：")
    print("1. 啟動後端服務器")
    print("2. 使用允許的 PID 登入系統")
    print("3. 通過 /api/admin/allowed-pids 管理 PID")
    print("4. 通過 /api/admin/chat-sessions 查看聊天會話")