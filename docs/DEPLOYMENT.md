# Deployment Guide - AI Trading Copilot

## 📋 Overview

This guide covers deploying the AI Trading Copilot system using Docker. The system consists of two main services:
- **Streamlit Application**: Web UI for trading, agents, and analytics
- **PostgreSQL Database**: Production-grade database for trade journal

---

## 🎯 Deployment Modes

### Development Mode (SQLite)
- **Database**: SQLite file at `data/trades.db`
- **Setup**: Zero configuration
- **Use case**: Local development, testing, single-user
- **Command**: `streamlit run streamlit_app/app.py`

### Production Mode (PostgreSQL + Docker)
- **Database**: PostgreSQL in Docker container
- **Setup**: Environment variables + Docker Compose
- **Use case**: Deployed app, multi-user, data persistence
- **Command**: `docker-compose up -d`

---

## 🚀 Quick Start (Docker Deployment)

### Prerequisites
- Docker 20.10+ installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose 2.0+ installed (included with Docker Desktop)
- 2GB free disk space
- HuggingFace API token ([Get one here](https://huggingface.co/settings/tokens))

### Step-by-Step Deployment

#### 1. Clone the Repository
```bash
git clone https://github.com/sohansputhran/ai-trading-copilot.git
cd ai-trading-copilot
```

#### 2. Configure Environment Variables
```bash
# Copy template to .env file
cp docker/.env.docker docker/.env

# Edit the file (use your preferred editor)
nano docker/.env  # or: code docker/.env, vim docker/.env, etc.
```

**Required changes in `docker/.env`:**
```bash
# MUST CHANGE: Use a strong password
POSTGRES_PASSWORD=your_strong_password_here

# MUST CHANGE: Add your HuggingFace token
HUGGINGFACE_API_TOKEN=hf_your_actual_token_here
```

**Optional changes:**
- `POSTGRES_USER`: Database username (default: `trading_user`)
- `POSTGRES_DB`: Database name (default: `trading_db`)
- `APP_PORT`: Streamlit port (default: `8501`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

#### 3. Start the Stack
```bash
# Navigate to docker directory
cd docker

# Start all services (detached mode)
docker-compose up -d
```

This command will:
1. Build the Streamlit application image (~3-5 minutes first time)
2. Pull PostgreSQL image (~1 minute)
3. Create and start both containers
4. Initialize PostgreSQL with schema from `src/execution/schema.sql`
5. Set up health checks and networking

#### 4. Verify Deployment
```bash
# Check service status
docker-compose ps

# Expected output:
# NAME                  STATUS              PORTS
# trading_app           Up (healthy)        0.0.0.0:8501->8501/tcp
# trading_postgres      Up (healthy)        0.0.0.0:5432->5432/tcp

# Check application logs
docker-compose logs -f app

# Check database logs
docker-compose logs -f postgres
```

#### 5. Access the Application
Open your browser and navigate to:
```
http://localhost:8501
```

You should see the AI Trading Copilot dashboard.

---

## 🔄 Migrating Existing SQLite Data

If you have existing trades in SQLite (from local development), migrate them to PostgreSQL:

### Step 1: Ensure PostgreSQL is Running
```bash
cd docker
docker-compose ps postgres
# Should show: Up (healthy)
```

### Step 2: Set DATABASE_URL
```bash
# Windows PowerShell
$env:DATABASE_URL="postgresql://trading_user:your_password@localhost:5432/trading_db"

# Linux/Mac
export DATABASE_URL="postgresql://trading_user:your_password@localhost:5432/trading_db"
```

**Important**: Replace `your_password` with the actual `POSTGRES_PASSWORD` from your `docker/.env` file.

### Step 3: Run Migration Script
```bash
# From project root directory
python scripts/migrate.py

# Or with explicit paths
python scripts/migrate.py --sqlite data/trades.db --postgres $DATABASE_URL

# Dry-run mode (preview without migrating)
python scripts/migrate.py --dry-run
```

**Expected output:**
```
2024-03-15 10:30:00 - INFO - Reading trades from SQLite: data/trades.db
2024-03-15 10:30:00 - INFO - Found 15 closed trades in SQLite
2024-03-15 10:30:01 - INFO - Migration complete: 15 inserted, 0 skipped
2024-03-15 10:30:01 - INFO - ✅ Verification passed: counts match
```

### Idempotency
The migration script is **idempotent** - safe to run multiple times:
- Existing trades are automatically skipped (no duplicates)
- Only new trades are inserted
- Partial migrations can be resumed safely

---

## 🛠️ Common Operations

### View Logs
```bash
# All services
docker-compose logs -f

# Just the app
docker-compose logs -f app

# Just the database
docker-compose logs -f postgres

# Last 100 lines
docker-compose logs --tail=100 app
```

### Restart Services
```bash
# Restart all services
docker-compose restart

# Restart just the app
docker-compose restart app

# Rebuild and restart (after code changes)
docker-compose up -d --build app
```

### Stop Services
```bash
# Stop all services (data persists)
docker-compose down

# Stop and remove volumes (⚠️ DELETES ALL DATA)
docker-compose down -v
```

### Database Access

#### PostgreSQL Command Line
```bash
# Connect to PostgreSQL from host
docker exec -it trading_postgres psql -U trading_user -d trading_db

# Common queries:
SELECT COUNT(*) FROM trades;
SELECT * FROM trades ORDER BY entry_timestamp DESC LIMIT 10;
\dt  # List tables
\d trades  # Describe trades table
\q  # Quit
```

#### Database Backup
```bash
# Backup to file
docker exec trading_postgres pg_dump -U trading_user trading_db > backup_$(date +%Y%m%d).sql

# Restore from backup
docker exec -i trading_postgres psql -U trading_user -d trading_db < backup_20240315.sql
```

### Check Resource Usage
```bash
# Container resource stats (live)
docker stats trading_app trading_postgres

# Disk usage
docker system df

# Detailed volume inspection
docker volume inspect docker_postgres_data
```

---

## 🔧 Troubleshooting

### Problem: Container won't start
**Symptoms**: `docker-compose ps` shows container as `Exited` or `Restarting`

**Solution**:
```bash
# Check logs for error messages
docker-compose logs app

# Common causes:
# 1. Missing HUGGINGFACE_API_TOKEN
#    → Add token to docker/.env
# 2. Port 8501 already in use
#    → Change APP_PORT in docker/.env or stop conflicting process
# 3. Invalid DATABASE_URL format
#    → Check postgres service is healthy first
```

### Problem: PostgreSQL won't start
**Symptoms**: `trading_postgres` container unhealthy

**Solution**:
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Common causes:
# 1. Port 5432 already in use
#    → Change POSTGRES_PORT in docker/.env
# 2. Volume corruption
#    → docker-compose down -v (⚠️ deletes data)
#    → docker-compose up -d
# 3. Insufficient memory
#    → Reduce POSTGRES_SHARED_BUFFERS in docker/.env
```

### Problem: "Can't connect to database"
**Symptoms**: App logs show connection errors

**Solution**:
```bash
# Verify PostgreSQL is healthy
docker-compose ps postgres
# Should show: Up (healthy)

# Check DATABASE_URL is correct
docker exec trading_app env | grep DATABASE_URL
# Should show: DATABASE_URL=postgresql://trading_user:...@postgres:5432/trading_db

# Restart app container
docker-compose restart app
```

### Problem: Migration script fails
**Symptoms**: `scripts/migrate.py` exits with error

**Solution**:
```bash
# Check SQLite file exists
ls -lh data/trades.db

# Verify PostgreSQL is accessible
docker exec trading_postgres psql -U trading_user -d trading_db -c "SELECT 1;"

# Check your DATABASE_URL includes password
echo $DATABASE_URL  # Should show full connection string

# Try dry-run mode for debugging
python scripts/migrate.py --dry-run
```

### Problem: "Out of memory" errors
**Symptoms**: Container restarts unexpectedly, OOM in logs

**Solution**:
```bash
# Add resource limits to docker-compose.yml
# Uncomment the deploy.resources sections

# For PostgreSQL:
deploy:
  resources:
    limits:
      memory: 512M

# For app:
deploy:
  resources:
    limits:
      memory: 2G
```

---

## 🌍 Environment Variables Reference

### Required Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `HUGGINGFACE_API_TOKEN` | HuggingFace API key for LLM agents | `hf_abc123...` |
| `POSTGRES_PASSWORD` | PostgreSQL database password | `strong_password_123` |

### Database Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `trading_user` | PostgreSQL username |
| `POSTGRES_DB` | `trading_db` | PostgreSQL database name |
| `POSTGRES_PORT` | `5432` | PostgreSQL port (host) |
| `DATABASE_URL` | Auto-generated | Full PostgreSQL connection string |

### Application Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `APP_PORT` | `8501` | Streamlit port (host) |
| `ENV` | `production` | Environment (`development`, `production`) |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Trading Parameters
| Variable | Default | Description |
|----------|---------|-------------|
| `PAPER_TRADING` | `true` | Enable paper trading mode |
| `MAX_POSITION_SIZE` | `0.05` | Max 5% of portfolio per trade |
| `MAX_DAILY_LOSS` | `0.02` | Max 2% daily loss limit |

---

## 🔒 Security Best Practices

### Secrets Management
- ✅ **DO**: Store secrets in `.env` files (gitignored)
- ✅ **DO**: Use strong, unique passwords for `POSTGRES_PASSWORD`
- ✅ **DO**: Rotate API tokens periodically
- ❌ **DON'T**: Commit `.env` files to Git
- ❌ **DON'T**: Share your `.env` file or include it in screenshots
- ❌ **DON'T**: Use default passwords in production

### Network Security
- Containers communicate via internal Docker network
- PostgreSQL port `5432` is exposed to host for debugging (optional)
- In production, consider using Docker secrets or vault solutions
- Use firewall rules to restrict who can access `5432` and `8501`

### Database Security
- PostgreSQL runs with non-root user inside container
- Application container runs as `appuser` (non-root)
- Volumes are owned by container users only
- Regular backups recommended for production data

---

## 📊 Monitoring & Health Checks

### Container Health
```bash
# Check health status
docker inspect trading_app | grep -A 5 Health
docker inspect trading_postgres | grep -A 5 Health

# Health endpoints
curl http://localhost:8501/_stcore/health  # Streamlit health
```

### Application Metrics
- **Logs**: Available via `docker-compose logs`
- **Streamlit metrics**: Built into Streamlit Cloud if deployed there
- **Database metrics**: Use `pg_stat_statements` for query performance

---

## 🚀 Production Deployment (Advanced)

### Cloud Deployment Options

#### AWS
- **ECS/Fargate**: Container orchestration
- **RDS PostgreSQL**: Managed database
- **Secrets Manager**: Secure secrets storage

#### Google Cloud Platform
- **Cloud Run**: Serverless containers
- **Cloud SQL**: Managed PostgreSQL
- **Secret Manager**: Secrets management

#### Azure
- **Container Instances**: Simple container hosting
- **Azure Database for PostgreSQL**: Managed DB
- **Key Vault**: Secrets management

### Environment-Specific Configs
```bash
# Create environment-specific env files
docker/.env.dev        # Development settings
docker/.env.staging    # Staging environment
docker/.env.prod       # Production environment

# Use specific env file
docker-compose --env-file docker/.env.prod up -d
```

### Scaling Considerations
- Use managed PostgreSQL (RDS, Cloud SQL) for production
- Consider Redis for caching market data
- Set up log aggregation (ELK stack, CloudWatch)
- Implement monitoring (Prometheus, Datadog)
- Use load balancer for multiple app instances

---

## 📝 Maintenance Tasks

### Regular Backups
```bash
# Daily backup script (add to cron)
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
docker exec trading_postgres pg_dump -U trading_user trading_db | gzip > $BACKUP_DIR/trading_db_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "trading_db_*.sql.gz" -mtime +7 -delete
```

### Log Rotation
```bash
# Docker handles log rotation automatically
# Configure in daemon.json:
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

### Database Maintenance
```bash
# Vacuum database (reclaim space)
docker exec trading_postgres psql -U trading_user -d trading_db -c "VACUUM ANALYZE;"

# Check database size
docker exec trading_postgres psql -U trading_user -d trading_db -c "
SELECT pg_size_pretty(pg_database_size('trading_db'));"
```

---

## ❓ FAQ

**Q: Can I run this on Windows?**  
A: Yes! Docker Desktop works on Windows 10/11. All commands work in PowerShell.

**Q: Do I need to stop SQLite before using PostgreSQL?**  
A: No, they're independent. SQLite is used when `DATABASE_URL` is unset, PostgreSQL when it's set.

**Q: How do I update the application?**  
A: `git pull origin main && cd docker && docker-compose up -d --build`

**Q: Can I run without Docker?**  
A: Yes, but you'll need to manually install/run PostgreSQL. Docker is recommended for consistency.

**Q: How much does this cost to run?**  
A: **Locally**: Free (just your electricity)  
**AWS**: ~$20-50/month (t3.small + RDS db.t3.micro)  
**GCP/Azure**: Similar pricing

**Q: Is this production-ready?**  
A: For personal use: Yes  
For business-critical trading: Add monitoring, automated backups, HA setup

---

## 📞 Support

- **GitHub Issues**: [github.com/sohansputhran/ai-trading-copilot/issues](https://github.com/sohansputhran/ai-trading-copilot/issues)
- **Documentation**: Check `README.md` and code comments
- **Logs**: Always check `docker-compose logs` first when debugging

---

**Happy Trading! 🚀**
