#!/bin/bash

echo "📌 Setting up swap memory..."
fallocate -l 1G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "✅ Swap memory enabled!"

# Start your application
gunicorn app:app
