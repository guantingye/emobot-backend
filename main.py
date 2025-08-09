# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from recommend import compute_recommendation

app = FastAPI(title="Emobot+ API", version="1.0.0")

# 從環境變數讀取前端 URL
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "https://*.vercel.app"],  # 支援 Vercel 部署
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecRequest(BaseModel):
    mbti: List[int]    # [0 or 1, 0 or 1, 0 or 1, 0 or 1]
    ders: List[float]  # length 18
    aas: List[float]   # length 24  
    bpns: List[float]  # length 21

@app.get("/")
async def root():
    return {"message": "Emobot+ API is running!", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    return {"status": "OK", "service": "Emobot+ API"}

@app.post("/api/recommend")
async def recommend(req: RecRequest):
    """
    根據心理測驗結果推薦最適合的 AI 類型
    """
    try:
        result = compute_recommendation(req.dict())
        
        # 轉換為前端期望的格式
        # 映射 AI 名稱到 ID
        ai_name_to_id = {
            "EmpathicAI": 1,           # 同理型 AI
            "InsightfulAI": 2,         # 洞察型 AI  
            "Solution-FocusedAI": 3,   # 解決型 AI
            "CognitiveAI": 4           # 認知型 AI
        }
        
        # 轉換 all_scores 格式
        all_scores = {}
        for item in result["all_scores"]:
            ai_name = item["label"]
            if ai_name in ai_name_to_id:
                all_scores[ai_name_to_id[ai_name]] = round(item["sim"], 3)
        
        return {
            "recommended_bot": ai_name_to_id.get(result["best"], 1),
            "confidence": result["confidence"], 
            "all_scores": all_scores,
            "details": {
                "ders_scores": result["ders_scores"],
                "aas_scores": result["aas_scores"], 
                "bpns_scores": result["bpns_scores"],
                "tsne": result["tsne"]
            }
        }
        
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/api/selected-bot")
async def selected_bot(request: dict):
    """
    記錄用戶選擇的機器人
    """
    bot_id = request.get("bot_id")
    return {"success": True, "selected_bot": bot_id}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        reload=False  # 生產環境關閉 reload
    )