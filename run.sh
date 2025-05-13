#!/bin/bash

# Skin Lesion Segmentation Project Script

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create output directories
mkdir -p outputs_fcn/models
mkdir -p outputs_fcn/results
mkdir -p outputs_deeplabv3/models
mkdir -p outputs_deeplabv3/results
mkdir -p comparison_results

# Train and evaluate FCN model
echo "Training and evaluating FCN model..."
python main.py --mode train --models fcn --epochs 50 --batch_size 32 --img_size 512 --lr 5e-5 --output_dir ./outputs_fcn --amp

# Train and evaluate DeepLabV3 model
echo "Training and evaluating DeepLabV3 model..."
python main.py --mode train --models deeplabv3 --epochs 50 --batch_size 32 --img_size 512 --lr 5e-5 --output_dir ./outputs_deeplabv3 --amp

# Compare the two models
echo "Comparing FCN and DeepLabV3 models..."
FCN_MODEL="./outputs_fcn/models/fcn_best.pth"
DEEPLABV3_MODEL="./outputs_deeplabv3/models/deeplabv3_best.pth"

if [ -f "$FCN_MODEL" ] && [ -f "$DEEPLABV3_MODEL" ]; then
    echo "Both models found, running comparison..."
    python evaluate.py --output_dir ./comparison_results --model fcn --model_path "$FCN_MODEL"
    python evaluate.py --output_dir ./comparison_results --model deeplabv3 --model_path "$DEEPLABV3_MODEL"
else
    echo "WARNING: One or both model files not found:"
    [ ! -f "$FCN_MODEL" ] && echo "- FCN model not found at $FCN_MODEL"
    [ ! -f "$DEEPLABV3_MODEL" ] && echo "- DeepLabV3 model not found at $DEEPLABV3_MODEL"
    echo "Please make sure training has completed successfully before running comparison."
fi

echo "Done!" 