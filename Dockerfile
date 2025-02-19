# Stage 1: Install dependencies in a lightweight environment
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final lightweight image
FROM python:3.12-slim AS final
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

CMD ["python", "app.py"]



