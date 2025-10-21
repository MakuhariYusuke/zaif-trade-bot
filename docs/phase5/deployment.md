# V433 Phase 5 デプロイメントガイド

## 概要

このガイドでは、V433 Phase 5 Production Migration Systemの本番環境へのデプロイ手順を説明します。

## 前提条件

### システム要件
- **OS**: Linux (Ubuntu 20.04+), Windows Server 2019+, macOS 11+
- **Python**: 3.11.0 以上
- **メモリ**: 最低 8GB, 推奨 16GB
- **ストレージ**: 最低 50GB SSD
- **ネットワーク**: 安定したインターネット接続

### インフラ要件
- **データベース**: PostgreSQL 13+ または MySQL 8.0+
- **Redis**: 6.0+ (オプション、キャッシュ用)
- **ロードバランサー**: Nginx または HAProxy
- **監視**: Prometheus + Grafana (推奨)

## デプロイスクリプト

### 1. 環境準備スクリプト

```bash
#!/bin/bash
# scripts/deploy/prepare_environment.sh

set -e

echo "=== V433 Phase 5 Environment Preparation ==="

# Python バージョン確認
python_version=$(python3 --version | cut -d' ' -f2)
required_version="3.11.0"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "Error: Python $required_version or higher required. Current: $python_version"
    exit 1
fi

# 仮想環境作成
echo "Creating virtual environment..."
python3 -m venv venv_prod
source venv_prod/bin/activate

# 依存関係インストール
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements_temp.txt

# 設定ファイル作成
echo "Creating configuration files..."
mkdir -p config/prod
cat > config/prod/app.yaml << EOF
environment: production
database:
  url: ${DATABASE_URL}
  pool_size: 20
  max_overflow: 30
redis:
  url: ${REDIS_URL:-}
  ttl: 3600
logging:
  level: INFO
  file: logs/production.log
monitoring:
  enabled: true
  metrics_port: 9090
emergency:
  circuit_breaker_enabled: true
  auto_recovery_enabled: true
EOF

echo "Environment preparation completed!"
```

### 2. デプロイスクリプト

```bash
#!/bin/bash
# scripts/deploy/deploy.sh

set -e

echo "=== V433 Phase 5 Production Deployment ==="

# 環境変数読み込み
if [ -f .env ]; then
    source .env
fi

# 事前チェック
echo "Running pre-deployment checks..."
python scripts/maintenance/health_check.py

# データベースマイグレーション
echo "Running database migrations..."
alembic upgrade head

# テスト実行
echo "Running integration tests..."
python tests/run_integration_tests.py

if [ $? -ne 0 ]; then
    echo "Integration tests failed! Aborting deployment."
    exit 1
fi

# アプリケーション起動
echo "Starting application..."
export PYTHONPATH=$(pwd):$PYTHONPATH

# Gunicornで起動（本番用）
gunicorn \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --log-level info \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    ztb.app:app &

echo $! > app.pid

# ヘルスチェック
sleep 10
curl -f http://localhost:8000/health || (echo "Health check failed!" && exit 1)

echo "Deployment completed successfully!"
echo "Application is running on http://localhost:8000"
```

### 3. ロールバックスクリプト

```bash
#!/bin/bash
# scripts/deploy/rollback.sh

set -e

echo "=== V433 Phase 5 Rollback ==="

# アプリケーション停止
if [ -f app.pid ]; then
    pid=$(cat app.pid)
    kill $pid 2>/dev/null || true
    rm app.pid
    echo "Application stopped"
fi

# 以前のバージョンに戻す
echo "Rolling back to previous version..."
git checkout HEAD~1
git submodule update --init --recursive

# 依存関係再インストール
source venv_prod/bin/activate
pip install -r requirements_temp.txt

# アプリケーション再起動
echo "Restarting application..."
bash scripts/deploy/deploy.sh

echo "Rollback completed!"
```

## Docker デプロイメント

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# システム依存関係
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係
COPY requirements_temp.txt .
RUN pip install --no-cache-dir -r requirements_temp.txt

# アプリケーションコード
COPY . .

# 設定
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 起動
CMD ["python", "-m", "uvicorn", "ztb.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  zaif-trading-v433:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/zaif_trading
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: zaif_trading
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

## Kubernetes デプロイメント

### デプロイメントマニフェスト

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zaif-trading-v433
spec:
  replicas: 3
  selector:
    matchLabels:
      app: zaif-trading-v433
  template:
    metadata:
      labels:
        app: zaif-trading-v433
    spec:
      containers:
      - name: zaif-trading
        image: zaif/trading:v433-phase5
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### サービスマニフェスト

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: zaif-trading-service
spec:
  selector:
    app: zaif-trading-v433
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 監視設定

### Prometheus 設定

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'zaif-trading'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

### Grafana ダッシュボード

主要なメトリクス：
- 取引量
- 成功率
- 応答時間
- エラー率
- システムリソース使用率
- 回路ブレーカー状態

## セキュリティ設定

### 環境変数の管理

```bash
# .env.production
DATABASE_URL=postgresql://user:encrypted_pass@host:5432/db
REDIS_URL=redis://host:6379
SECRET_KEY=your-secret-key-here
API_KEYS=comma,separated,keys
```

### ファイアウォール設定

```bash
# UFW設定例
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 8000/tcp    # Application
sudo ufw --force enable
```

## パフォーマンスチューニング

### Gunicorn 設定

```python
# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30
keepalive = 10
```

### データベース最適化

```sql
-- PostgreSQL設定
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

## バックアップと復旧

### 自動バックアップスクリプト

```bash
#!/bin/bash
# scripts/backup/backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/$DATE"

mkdir -p $BACKUP_DIR

# データベースバックアップ
pg_dump zaif_trading > $BACKUP_DIR/database.sql

# 設定ファイルバックアップ
cp -r config/ $BACKUP_DIR/

# ログアーカイブ
tar -czf $BACKUP_DIR/logs.tar.gz logs/

# 古いバックアップ削除（7日以上前）
find /backups -type d -mtime +7 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR"
```

## トラブルシューティング

### デプロイ失敗時の対応

1. **ログ確認**
   ```bash
   tail -f logs/production.log
   docker logs zaif-trading-v433
   kubectl logs -l app=zaif-trading-v433
   ```

2. **ヘルスチェック**
   ```bash
   curl -f http://localhost:8000/health
   ```

3. **リソース確認**
   ```bash
   top
   df -h
   docker stats
   ```

4. **ロールバック**
   ```bash
   bash scripts/deploy/rollback.sh
   ```

### パフォーマンス問題

- CPU使用率が高い場合: ワーカー数を増やす
- メモリ使用率が高い場合: インスタンスを増やす
- 応答時間が遅い場合: データベースクエリを最適化

## メンテナンス

### 定期メンテナンス

- **日次**: ログローテーション、ヘルスチェック
- **週次**: セキュリティアップデート、バックアップ検証
- **月次**: パフォーマンスレポート、依存関係更新

### 更新手順

1. 新バージョンリリース
2. ステージング環境でテスト
3. カナリアデプロイメント
4. 本番環境更新
5. 監視と検証

---

このガイドはV433 Phase 5の安全な本番デプロイを支援するためのものです。実際のデプロイ前にステージング環境で十分なテストを行ってください。