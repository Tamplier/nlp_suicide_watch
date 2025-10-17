#!/bin/bash
set -e

if [ "$APP_ENV" = "prod" ]; then
    if [ "$CLIENT" = "telegram" ]; then
        echo "Starting telegram bot..."
        exec invoke start-telegram-bot
    elif [ "$CLIENT" = "gradio" ]; then
        echo "Starting gradio..."
        exec invoke gradio
    else
        echo "Starting cli..."
        exec invoke cli
    fi
else
    echo "Test build finished"
fi
