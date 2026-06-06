#!/bin/bash

echo "🚀 部署TA-Lib技术指标计算服务..."

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请先启动Docker"
    exit 1
fi

# 停止现有容器
echo "🛑 停止现有容器..."
docker-compose down

# 拉取最新代码（部署机以远端为准，丢弃本地未提交改动，避免 pull 被 deploy.sh 等脏文件挡住）
echo "📥 拉取最新代码..."
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if ! git fetch origin "$BRANCH"; then
    echo "❌ git fetch 失败，已中止部署"
    exit 1
fi
git reset --hard "origin/$BRANCH"

# 清理旧镜像（可选）
echo "🧹 清理旧镜像..."
docker system prune -f

# 构建并启动
echo "🔨 构建并启动服务..."
docker-compose up --build -d

# 等待启动
echo "⏳ 等待服务启动..."
sleep 15

# 健康检查
echo "🏥 检查服务状态..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ 服务启动成功！"
    echo ""
    echo "📖 API文档: http://localhost:8000/docs"
    echo "🏥 健康检查: http://localhost:8000/health"
    echo "🔧 技术指标: http://localhost:8000/calculate"
    echo "🖼️ 图片生成: http://localhost:8000/generate"
else
    echo "❌ 服务启动失败，请检查日志："
    echo "docker-compose logs"
fi
