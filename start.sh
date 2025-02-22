#!/bin/bash

echo "📌 Setting up swap memory..."
fallocate -l 1G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "✅ Swap memory enabled!"

# Start Flask app with Gunicorn
gunicorn app:app --bind 0.0.0.0:10000 --workers=2

