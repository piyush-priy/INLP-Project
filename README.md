# CARR: Capability-Aware Residual Routing

CARR (Capability-Aware Residual Routing) is an efficient routing architecture designed to enhance the representation capacity of Large Language Models (LLMs) by integrating dynamic, learnable gating mechanisms over a continuous residual stream. 

By replacing dense components with a Saliency-Aware Decentralized Mixture-of-Experts (SAD-MoE) inspired architecture, CARR drastically reduces perplexity while only requiring ~0.45% trainable parameters to be calibrated on a frozen baseline model.

## Features

- **Efficient MoE Dispatch**: Augments specific layers (such as V-projections) into a lightweight Mixture-of-Experts architecture on top of a frozen backbone.
- **Residual Gate Blending**: Learns to balance between the original frozen pre-trained weights and the calibrated routing network through a dynamic, learnable `alpha`.
- **Shared Expert Architecture**: Reserves a specific expert stream as a generalized "catch-all" to improve routing adaptability and model generalization, outperforming standard Top-K routing by mitigating expert starvation.
- **Fast Calibration**: Designed to calibrate large scale models (e.g., Mixtral 8x7B) on accessible consumer hardware using 4-bit quantization (BitsAndBytes) and gradient accumulation.

## Repository Structure

```text
├── carr/                      # Core model library
│   ├── models/                # Model patching logic (e.g., mixtral_carr.py)
│   ├── trainer/               # Calibrator loop (calibrator.py)
│   └── utils/                 # Data loading, metrics, and visualization utilities
├── configs/                   # YAML configurations for various experiment modes
├── presentation/              # Typst source for the presentation of results
├── report/                    # LaTeX source for the final ACL-format research paper
├── scripts/                   # Evaluation and evaluation scripts pipeline
│   ├── run_all_modes.py       # Wrapper to run multi-mode comparative experiments
│   ├── plot_comparison.py     # Generates visualizations from experiment results
│   ├── run_calibrate.py       # Standalone script for calibration
│   └── run_eval.py            # Standalone evaluation script
├── colab_quickstart.py        # Interactive single-file evaluation script
└── analysis_results.md        # Deep dive into experimental findings and routing dynamics
```

## Getting Started

### Prerequisites

You need a CUDA-capable GPU for efficient training and inference. Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch>=2.1.0`
- `transformers>=4.36.0`
- `bitsandbytes>=0.41.0`
- `accelerate>=0.25.0`
- `datasets>=2.16.0`

### Quickstart (Colab / Single File)

To quickly smoke-test the calibration pipeline and see perplexity gains on a 4-bit quantized model (Mistral or Mixtral):

```bash
python colab_quickstart.py
```

*Tip: Use the `--debug` flag for a fast end-to-end dataset truncation check (1 epoch, 1k tokens, seq_len=128) to ensure your environment is set up correctly.*

## Training & Experiment Pipeline

The project relies heavily on the `scripts/run_all_modes.py` entrypoint to consistently benchmark the various theoretical modes. The configuration system uses YAML files found in `configs/`.

### Configuration Modes

The pipeline allows comparison across 4 main configurations:
1. **Mode 1 - Baseline** (`configs/mode1_baseline.yaml`): Uncalibrated, frozen base model behavior. Used as the negative control.
2. **Mode 2 - Gate-Only** (`configs/mode2_gate_only.yaml`): Updates only expert routing parameters while leaving the residual path inactive ($α \approx 0$).
3. **Mode 3 - Full CARR** (`configs/mode3_full_carr.yaml`): Complete residual routing which actively updates the gate and dynamic capability scalars. 
4. **Mode 4 - Shared Expert** (`configs/mode4_shared_expert.yaml`): The premier configuration, dedicating one persistent expert to capture dense capabilities, reducing token dropping errors.

### Running the Comparative Study

You can run an individual mode or all modes sequentially to reproduce the research results:

```bash
# Run the entire 4-mode benchmarking suite
python scripts/run_all_modes.py --results_dir ./carr_output

# Run a specific mode
python scripts/run_all_modes.py --mode shared_expert --results_dir ./carr_output

# Enable debug mode to test the pipeline rapidly
python scripts/run_all_modes.py --debug
```

### Plotting Experimental Results

Once your experiments output their loss and JSON histories to the `--results_dir` (e.g., `./carr_output`), you can generate publication-ready plots:

```bash
python scripts/plot_comparison.py --results_dir ./carr_output
```
This script will produce artifacts such as perplexity drops across epochs, heatmap distributions of expert usage (Load Entropy), and comparative routing uniqueness (Jaccard Index). These visuals are heavily featured in the `report/` and `presentation/` directories.

## Key Results

After a 3-epoch calibration utilizing only ~17M parameters (~0.45% of total trainable model parameters), CARR demonstrates a transformative perplexity improvement from a randomized baseline:

| Mode | Perplexity ↓ | Load Entropy (H) ↑ | CoV ↓ | Jaccard ↓ | Total Wall Time |
|---|---|---|---|---|---|
| **Baseline** | 570.59 | N/A | N/A | N/A | 97.4s |
| **Gate-Only** | 23.04 | 2.463 | 0.931 | 0.070 | 854.6s |
| **Full CARR** | 21.79 | **2.516** | **0.880** | 0.073 | 854.4s |
| **Shared Expert**| **16.58** | 2.334 | 1.004 | **0.064** | **850.1s** |

#### Main Takeaways:
1. **Best Generalization**: The **Shared Expert** approach reliably outpaces the other variations by achieving a 28% perplexity improvement over standard gate learning.
2. **Highly Hardware Efficient**: The entire robust training benchmark dynamically scales efficiently with Mixed Precision & Gradient Accumulation, completing in under 15 minutes (~850s).
3. **Optimized Load Balancing**: The system achieves an incredibly balanced routing coefficient of variation (CoV), ensuring that most specialized experts are saturated equitably. 

For more deep-dive analyses, ablation graphs, and layer-wise $α$ findings, explore the comprehensive [analysis_results.md](analysis_results.md).
