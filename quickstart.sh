#!/bin/bash

set -e

echo "===================="
echo ""
echo "="*80
# Setup
echo "Setting up environment..."
make setup

echo ""
echo "Syncing dependencies..."
make sync

echo ""
echo "Preprocessing data..."
make preprocess

echo ""
echo "Running smoke test..."
make smoke

echo ""
echo "Training critic model..."
make train_critic

echo ""
echo "Fine-tuning doctor model..."
make sft_doctor

echo ""