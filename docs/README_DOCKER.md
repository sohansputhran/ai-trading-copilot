# Docker Deployment (Sprint 6)

## Quick Start with Docker

The easiest way to run the AI Trading Copilot is using Docker:

```bash
# 1. Clone the repository
git clone https://github.com/sohansputhran/ai-trading-copilot.git
cd ai-trading-copilot

# 2. Configure environment
cp docker/.env.docker docker/.env
# Edit docker/.env and add your HUGGINGFACE_API_TOKEN and POSTGRES_PASSWORD

# 3. Start the stack
cd docker
docker-compose up -d

# 4. Access the application
# Open http://localhost:8501 in your browser
```

**That's it!** The application and PostgreSQL database are now running in containers.

## What Gets Deployed

Docker Compose starts two services:
- **trading_app**: Streamlit dashboard (port 8501)
- **trading_postgres**: PostgreSQL 16 database (port 5432)

Both services include health checks and auto-restart on failure.

## Deployment Modes

### Development Mode (SQLite)
```bash
# Run locally without Docker
streamlit run streamlit_app/app.py
```
Uses SQLite at `data/trades.db` — zero configuration needed.

### Production Mode (PostgreSQL + Docker)
```bash
# Run with Docker Compose
cd docker && docker-compose up -d
```
Uses PostgreSQL — production-ready, multi-user support.

## Migrating Data

Have existing trades in SQLite? Migrate them to PostgreSQL:

```bash
# Set DATABASE_URL to your PostgreSQL instance
export DATABASE_URL="postgresql://trading_user:password@localhost:5432/trading_db"

# Run migration script (idempotent - safe to run multiple times)
python scripts/migrate.py
```

## Common Commands

```bash
# View logs
docker-compose logs -f app

# Restart services
docker-compose restart

# Stop everything (data persists)
docker-compose down

# Stop and delete data
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build
```

## Full Documentation

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for comprehensive deployment guide including:
- Environment variable reference
- Cloud deployment options (AWS, GCP, Azure)
- Troubleshooting guide
- Security best practices
- Backup and maintenance procedures

## Requirements

- Docker 20.10+ ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose 2.0+ (included with Docker Desktop)
- 2GB free disk space
- HuggingFace API token ([Get token](https://huggingface.co/settings/tokens))
