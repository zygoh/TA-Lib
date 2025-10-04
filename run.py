#!/usr/bin/env python3
"""
TA-Lib API 服务启动脚本
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0", 
        port=8000,
        workers=1,
        log_level="info",
        access_log=True
    )
