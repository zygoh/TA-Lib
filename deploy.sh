#!/bin/bash

echo "🚀 部署技术指标计算服务..."

# 停止现有容器
docker-compose down

# 构建并启动
docker-compose up --build -d

# 等待启动
sleep 10

echo "✅ 服务已启动"
echo "📖 API文档: http://localhost:8000/docs"
echo "🏥 健康检查: http://localhost:8000/health"
