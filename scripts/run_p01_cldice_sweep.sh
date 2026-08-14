#!/bin/bash
# P0-1: clDice Sensitivity Study
# Sweep clDice weight for E1 (DW) and E4 (SIP-v2 Min)

cd /root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular

# Fixed conditions
SEED=42
EPOCHS=200
BS=2
SIZE=512
WARMUP=20

# clDice weights to sweep
LAMBDAS=(0.0 0.1 0.2 0.3 0.5 1.0)

# E1 (DW) sweep
for LAMBDA in "${LAMBDAS[@]}"; do
    echo "=== E1 + clDice lambda=$LAMBDA ==="
    if [ "$LAMBDA" = "0.0" ]; then
        # No clDice
        python scripts/train.py \
            --model E1 \
            --dataset drive \
            --seed $SEED \
            --epochs $EPOCHS \
            --batch_size $BS \
            --img_size $SIZE \
            --output_dir outputs/p01_e1_lambda${LAMBDA}_seed${SEED}
    else
        python scripts/train.py \
            --model E1 \
            --dataset drive \
            --seed $SEED \
            --epochs $EPOCHS \
            --batch_size $BS \
            --img_size $SIZE \
            --use_cldice \
            --cldice_lambda $LAMBDA \
            --cldice_warmup $WARMUP \
            --output_dir outputs/p01_e1_lambda${LAMBDA}_seed${SEED}
    fi
done

# E4 (SIP-v2 Min) sweep
for LAMBDA in "${LAMBDAS[@]}"; do
    echo "=== E4 + clDice lambda=$LAMBDA ==="
    if [ "$LAMBDA" = "0.0" ]; then
        python scripts/train.py \
            --model E4 \
            --dataset drive \
            --seed $SEED \
            --epochs $EPOCHS \
            --batch_size $BS \
            --img_size $SIZE \
            --output_dir outputs/p01_e4_lambda${LAMBDA}_seed${SEED}
    else
        python scripts/train.py \
            --model E4 \
            --dataset drive \
            --seed $SEED \
            --epochs $EPOCHS \
            --batch_size $BS \
            --img_size $SIZE \
            --use_cldice \
            --cldice_lambda $LAMBDA \
            --cldice_warmup $WARMUP \
            --output_dir outputs/p01_e4_lambda${LAMBDA}_seed${SEED}
    fi
done

echo "P0-1 clDice sweep complete!"
