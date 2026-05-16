#!/bin/bash
# Deploy to all tenant Railway services
# Requires Railway CLI: npm install -g @railway/cli

echo "Deploying to demo..."
railway redeploy --service savant-demo

echo "Deploying to innovim..."
railway redeploy --service savant-innovim

echo "Deploying to potencia..."
railway redeploy --service savant-potencia

echo "All deployments triggered."
