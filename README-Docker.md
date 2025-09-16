# 技术指标计算服务 Docker 部署指南

## 🐳 Docker 部署

### 快速开始

```bash
# 1. 克隆项目
git clone <your-repo>
cd TA-Lib

# 2. 运行部署脚本
chmod +x deploy.sh
./deploy.sh
```

### 手动部署

```bash
# 1. 构建镜像
docker-compose build

# 2. 启动服务
docker-compose up -d

# 3. 查看日志
docker-compose logs -f

# 4. 停止服务
docker-compose down
```

## 📁 文件说明

- `Dockerfile` - Docker镜像构建文件
- `docker-compose.yml` - 开发环境编排文件
- `docker-compose.prod.yml` - 生产环境编排文件
- `docker-config.json` - Docker环境配置文件
- `nginx.conf` - Nginx反向代理配置
- `deploy.sh` - 自动部署脚本
- `.dockerignore` - Docker忽略文件

## 🔧 环境配置

### 开发环境
```bash
docker-compose up -d
```

### 生产环境
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📊 服务访问

- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **API接口**: http://localhost:8000/calculate

## 🛠️ 常用命令

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 重新构建
docker-compose build --no-cache

# 进入容器
docker-compose exec ta-lib-api bash
```

## 📈 性能优化

### 生产环境配置
- 使用多进程模式
- 配置资源限制
- 启用Nginx反向代理
- 设置健康检查

### 监控建议
- 监控内存使用
- 监控CPU使用
- 监控API响应时间
- 设置日志轮转

## 🔍 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 修改docker-compose.yml中的端口映射
   ports:
     - "8001:8000"  # 改为其他端口
   ```

2. **内存不足**
   ```bash
   # 增加Docker内存限制
   deploy:
     resources:
       limits:
         memory: 2G
   ```

3. **TA-Lib安装失败**
   ```bash
   # 检查系统依赖
   docker-compose exec ta-lib-api python -c "import talib"
   ```

### 日志查看
```bash
# 查看所有日志
docker-compose logs

# 查看特定服务日志
docker-compose logs ta-lib-api

# 实时查看日志
docker-compose logs -f ta-lib-api
```

## 🚀 扩展部署

### 多实例部署
```yaml
# 修改docker-compose.prod.yml
services:
  ta-lib-api:
    deploy:
      replicas: 3
```

### 负载均衡
```yaml
# 使用Nginx负载均衡
upstream ta_lib_api {
    server ta-lib-api-1:8000;
    server ta-lib-api-2:8000;
    server ta-lib-api-3:8000;
}
```

## 📝 注意事项

1. **数据持久化**: 日志文件挂载到宿主机
2. **配置管理**: 使用配置文件挂载
3. **安全设置**: 生产环境建议使用非root用户
4. **资源监控**: 定期检查容器资源使用情况
5. **备份策略**: 定期备份配置和日志文件
