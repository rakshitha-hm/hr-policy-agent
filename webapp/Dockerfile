FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /web
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd --create-home webuser
USER webuser
EXPOSE 5000
# gunicorn: production web server (Flask's built-in server is for development only)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "180", "app:app"]