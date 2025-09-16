#!/bin/bash
set -e

# 等待服务启动
echo "等待服务启动..."
sleep 5

# 检查健康状态
echo "检查服务健康状态..."
curl -f http://localhost:8000/health || exit 1

echo "服务启动成功！"
echo "API文档地址: http://localhost:8000/docs"
echo "健康检查地址: http://localhost:8000/health"

# 启动主服务
exec "$@"
