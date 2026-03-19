## Compression Strategy
- Architecture Modification
- Knowledge Distillation
- Weight Pruning
- Fine-Tuning
- INT8 Quantization

The baseline MobileNetV1 was trained and used as a teacher for knowledge distillation.
A smaller student model was created by removing three redundant depthwise separable
layers (layers 10–12), reducing MACs from 7.49M to 5.60M. The student was trained using
a 70/30 split of soft teacher labels and hard ground-truth labels. Unstructured weight pruning
was then applied at 50% sparsity, followed by fine-tuning to recover lost accuracy. Finally,
INT8 post-training quantization was applied with 2000 calibration images, reducing model size
from 0.826MB to 0.220MB.

Why did some strategies not work?
1.Quantization-Aware Training (QAT) failed due to a version incompatibility between tfmot 0.7.3
and TensorFlow 2.15 — the NoOpActivation layer could not be deserialized by Keras.
2.Unstructured pruning had minimal impact because the model was already extremely small (alpha=0.25).
Removing 50% of individual weights from a model with so few parameters reduced capacity without meaningful
size or latency benefits on Raspberry Pi CPU, since the it lacks specialized sparse matrix hardware
to take advantage of weight sparsity.

Final Local metrics(test_public):
Accuracy = 83.43%
Latency = 3.31 ms
Model size = 0.220 MB
Peak memory = 358 MB
MACs = 5.60 M

Reproducing the Export Process
1.Environment Setup
conda activate vww_env
module load cuda12.2/toolkit/12.2.2
module load cudnn8.9-cuda12.2/8.9.7.29
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/vww_env/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$(dirname $(dirname $(which nvcc)))
2.Training Pipeline
# 1. Train baseline teacher model
python src/train_vww.py

# 2. Train student model with knowledge distillation
python src/train_kid.py

# 3. Apply unstructured pruning
python src/train_pruning.py

# 4. Fine-tune pruned model
python src/finetune_pruned.py

# 5. Convert to INT8 TFLite with 2000 calibration images
python src/convert_int8.py

# 6. Export final metrics JSON
python src/evaluate_vww.py --model trained_models/vww_96_small_kd_pruned_ft_int8.tflite --split test_public --compute_score --export_json
