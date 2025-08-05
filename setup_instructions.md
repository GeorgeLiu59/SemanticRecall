# Setup Instructions for Semantic Video Understanding System

## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n semantic_video python=3.9 -y
```

### 2. Activate Environment
```bash
# If conda init hasn't been run:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate semantic_video

# Or if conda init has been run:
conda activate semantic_video
```

### 3. Install PyTorch (CPU version)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Other Dependencies
```bash
pip install transformers clip-by-openai opencv-python scikit-learn scipy matplotlib seaborn tqdm moviepy faiss-cpu huggingface-hub datasets accelerate wandb pyyaml
```

### 5. Fix NumPy Version (if needed)
```bash
pip install "numpy<2"
```

## Usage

### Quick Test
```bash
python test_system.py
```

### Full Experiment
```bash
python run_experiment.py --video Bee_Movie_Cropped.mp4 --clusters 8 --experts 8
```

### Basic Usage
```python
from semantic_video import SemanticVideoProcessor

processor = SemanticVideoProcessor("video.mp4", num_clusters=10)
results = processor.process_video()
scenes = processor.query_semantic_memory("person walking")
```

## System Status

✅ **Environment Created**: `semantic_video` conda environment
✅ **Dependencies Installed**: All required packages installed
✅ **System Tested**: Basic functionality working
✅ **Video Processing**: Successfully processed 106 scenes from test video
✅ **Semantic Clustering**: Created 4 clusters with good metrics
✅ **Memory Hierarchy**: Built with 4 slots containing 106 scenes

## Next Steps

1. **Run Full Experiment**: Use `python run_experiment.py` with your video
2. **Customize Parameters**: Edit `configs/default_config.yaml`
3. **Analyze Results**: Check `results/` directory for outputs
4. **Visualize**: Generate plots with `processor.visualize_results()`

## Troubleshooting

- **CUDA Warning**: Normal if no GPU available, system will use CPU
- **NumPy Conflicts**: Use `pip install "numpy<2"` if needed
- **Memory Issues**: Reduce `num_clusters` or `memory_size` for large videos 