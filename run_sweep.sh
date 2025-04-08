#!/bin/bash

SWEEP_OUTPUT=$(wandb sweep sweep_config.yaml 2>&1)

CMD=$(echo "$SWEEP_OUTPUT" | awk -F'Run sweep agent with: ' '/Run sweep agent with:/ {print $2}')

echo "Running $CMD"

eval "$CMD"