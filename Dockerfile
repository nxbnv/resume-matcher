# 🚀 Stage 1: Install dependencies in a lightweight environment
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt . 

# ✅ Install dependencies in a separate directory (avoids unnecessary system installs)
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 🚀 Stage 2: Final minimal image
FROM python:3.12-slim AS final
WORKDIR /app

# ✅ Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# ✅ Copy the rest of the app
COPY . .

# ✅ Expose the correct port (important for Railway)
EXPOSE 5000

# ✅ Run Flask app dynamically using Railway's assigned port
CMD ["python", "app.py"]




