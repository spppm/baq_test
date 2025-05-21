FROM python:3.10-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

WORKDIR /app

# สร้าง virtual environment ที่ /app/.venv ก่อน
RUN python -m venv /app/.venv

# ตั้งค่า environment variable บอก uv ใช้ venv นี้
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# ติดตั้ง dependencies ผ่าน uv sync
RUN uv sync --frozen --no-cache

# รันแอป
CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--port", "80", "--host", "0.0.0.0"]
