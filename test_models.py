# test_models.py
"""
測試模型載入是否正常
執行方式：python test_models.py
"""

import os
import sys

# 添加專案根目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 載入環境變數
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def test_model_imports():
    """測試模型導入"""
    print("🔍 測試模型導入...")
    
    try:
        # 1. 測試基礎模組
        print("1. 導入基礎模組...")
        from app.db.base import Base
        from app.db.session import engine
        print("   ✅ 基礎模組導入成功")
        
        # 2. 測試核心模型
        print("2. 導入核心模型...")
        from app.models.user import User
        print("   ✅ User 模型導入成功")
        
        from app.models.assessment import Assessment
        print("   ✅ Assessment 模型導入成功")
        
        from app.models.recommendation import Recommendation
        print("   ✅ Recommendation 模型導入成功")
        
        from app.models.chat import ChatMessage
        print("   ✅ ChatMessage 模型導入成功")
        
        from app.models.mood import MoodRecord
        print("   ✅ MoodRecord 模型導入成功")
        
        # 3. 測試新模型
        print("3. 導入新模型...")
        from app.models.allowed_pid import AllowedPid
        print("   ✅ AllowedPid 模型導入成功")
        
        from app.models.chat_session import ChatSession
        print("   ✅ ChatSession 模型導入成功")
        
        # 4. 測試模型關聯
        print("4. 測試模型關聯...")
        
        # 檢查 User 模型的關聯
        user_relationships = [attr for attr in dir(User) if not attr.startswith('_')]
        expected_relationships = ['chat_messages', 'assessments', 'recommendations', 'moods', 'chat_sessions']
        
        for rel in expected_relationships:
            if hasattr(User, rel):
                print(f"   ✅ User.{rel} 關聯存在")
            else:
                print(f"   ⚠️  User.{rel} 關聯缺失")
        
        # 5. 測試表格創建
        print("5. 測試表格創建...")
        Base.metadata.create_all(bind=engine)
        print("   ✅ 表格創建成功")
        
        print("\n🎉 所有模型測試通過！")
        return True
        
    except Exception as e:
        print(f"\n❌ 模型測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_operations():
    """測試基本資料庫操作"""
    print("\n🔍 測試資料庫操作...")
    
    try:
        from sqlalchemy.orm import sessionmaker
        from app.db.session import engine
        from app.models.allowed_pid import AllowedPid
        
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        
        try:
            # 測試查詢操作
            count = db.query(AllowedPid).count()
            print(f"   📊 allowed_pids 表當前記錄數: {count}")
            
            # 測試插入操作（如果不存在的話）
            test_pid = "TEST1"
            existing = db.query(AllowedPid).filter(AllowedPid.pid == test_pid).first()
            
            if not existing:
                test_record = AllowedPid(
                    pid=test_pid,
                    description="測試記錄",
                    is_active=True
                )
                db.add(test_record)
                db.commit()
                print(f"   ✅ 成功新增測試記錄: {test_pid}")
                
                # 刪除測試記錄
                db.delete(test_record)
                db.commit()
                print(f"   ✅ 成功刪除測試記錄: {test_pid}")
            else:
                print(f"   ℹ️  測試記錄已存在: {test_pid}")
            
            print("   ✅ 資料庫操作測試成功")
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"   ❌ 資料庫操作測試失敗: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Emobot+ 模型測試工具")
    print("=" * 50)
    
    # 測試模型導入
    models_ok = test_model_imports()
    
    if models_ok:
        # 測試資料庫操作
        db_ok = test_database_operations()
        
        if db_ok:
            print("\n✅ 所有測試通過！可以安全執行遷移腳本。")
        else:
            print("\n⚠️  模型導入正常，但資料庫操作有問題。")
    else:
        print("\n❌ 模型載入失敗，請先修復模型定義問題。")
    
    print("\n建議：")
    print("1. 如果模型測試通過，可以執行: python migration_script.py")
    print("2. 如果有問題，請檢查模型文件是否正確創建")
    print("3. 確保所有必要的模型文件都已創建並放在正確位置")