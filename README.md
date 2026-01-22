# TerraComply

TerraComply is a web application built with Django REST Framework (backend) and Next.js (frontend) for compliance management and automation.

## Features
- RESTful API using Django REST Framework
- Modern frontend with Next.js and TypeScript
- Modular component-based architecture
- SQLite database (default)

## Project Structure

```
backend/
    manage.py
    core/           # Django project settings
    api/            # Django app with models, views, serializers
frontend/
    app/            # Next.js app directory
    components/     # React components
    hooks/          # Custom React hooks
    lib/            # Utility functions
```

## Getting Started

### Backend (Django)

1. **Install dependencies:**
   ```bash
   
   pip install -r requirements.txt
   ```
2. **Apply migrations:**
   ```bash
   cd crewAI
   ```
3. **Run the development server:**
   ```bash
  uvicorn main2:app --reload --port 8000
  
   ```

### Frontend (Next.js)

1. **Install dependencies:**
   ```bash
   cd frontend
   pnpm install
   ```
2. **Run the development server:**
   ```bash
   pnpm dev
   ```

## API Endpoints
- All API endpoints are available under `/api/` (see `backend/api/urls.py`).

## Environment Variables
- Configure Django settings in `backend/core/settings.py`.
- Configure Next.js environment in `frontend/.env` (if needed).

## License
MIT
