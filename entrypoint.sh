#!/bin/bash
set -e

if [ "$APP_ENV" = "prod" ]; then
    echo "Starting telegram bot..."
    exec invoke start-telegram-bot
else
    echo "Test build finished"
fi
