# Productionizing the Chess AI

Our goal is to transition our basic prototype into a fully trained, GUI-compatible chess engine using a $300 GCP compute budget. To achieve this, we need to accomplish three major milestones: (1) GUI Integration, (2) Dataset Selection, and (3) GCP Training Pipeline.

## 1. Connecting to a GUI (Arena / Lichess)

Chess GUIs like [Arena Chess](http://www.playwitharena.de/) or [CuteChess](https://cutechess.com/) do not know how python code works natively. They communicate with engines using a standard text-based protocol called **UCI (Universal Chess Interface)**.

**Implementation Steps:**
- Create `uci.py` as the main entry point for our engine.
- This script will listen to standard input (`sys.stdin.readline()`) for specific UCI commands sent by the GUI (e.g., `uci`, `isready`, `position startpos moves e2e4`, `go depth 3`).
- When our script receives `go`, it will trigger our `get_best_move_with_ai()` function from Phase 4, wait for the neural network to evaluate the tree, and output `bestmove e2e4` back to standard output (`sys.stdout.write()`).
- In Arena, you will simply configure a new engine, point it to your python executable, and pass `uci.py` as the argument.

## 2. The Golden Dataset

Training a neural network to evaluate chess positions requires millions of board states (FENs) matched with highly accurate evaluations (centipawns) from an existing super-engine like Stockfish.

**Recommended Dataset:**
We will use the **r2dev2/ChessData** (also available on HuggingFace as `ssingh22/chess-evaluations`). 
- **Size:** ~16 Million unique FEN positions.
- **Quality:** Every position has been evaluated by Stockfish 11 at a deep search depth of 22. 
- **Relevance:** It contains both tactical and random positions derived from real games, ensuring our network learns both quiet positional nuances and sharp tactics.
- **Format:** FEN string -> Centipawn evaluation. This is perfectly compatible with our `data_processing.py` script.

## 3. GCP Training Pipeline & Infrastructure

With a $300 budget, we can afford heavy firepower for a few days to churn through those 16 million positions.

### Machine Selection
- **Instance Type:** `g2-standard-4` or `g2-standard-8` on Google Cloud Compute Engine.
- **GPU:** **1x NVIDIA L4 GPU** (24GB VRAM). The L4 is arguably the most cost-effective architecture on GCP for pure neural network floating-point math right now. It costs roughly ~$0.60 to ~$0.80 per hour depending on region, meaning a 48-hour continuous training run will cost less than $40! 
- **OS/Disk:** Ubuntu 22.04 Deep Learning VM Image (comes pre-installed with PyTorch and NVIDIA CUDA drivers), backed by a 200GB SSD to hold the dataset.

### Code Adjustments for Production
We need to modify `train.py` before uploading it to the VM:
1. **Dataloader:** We will write a PyTorch `Dataset` class that reads the 16 million FEN strings from CSV/JSON in batches of 4096 to keep the GPU fed without running out of RAM.
2. **GPU Acceleration:** Add `.to('cuda')` to move the model and tensors off the CPU and onto the NVIDIA L4.
3. **Checkpointing:** The script must save `model_epoch_1.pth`, `model_epoch_2.pth`, etc., just in case the VM crashes midway.

### Logging (WandB / TensorBoard)
We cannot just rely on `print()` statements for a 2-day training run. 
- We will integrate **Weights & Biases (wandb)** into `train.py`.
- `wandb` will track our Loss curve, learning rate, and GPU temperature/utilization in real-time.
- You will be able to monitor the training progress on a beautiful web dashboard from your phone while the GCP server does the heavy lifting.

## Verification Plan

1. **Local UCI Test:** We will run `uci.py` locally in the terminal, type `position startpos` and `go depth 2`, and verify the engine responds with a formatted `bestmove`.
2. **GUI Hookup:** Connect the local, untrained engine to Arena Chess to verify the visual pieces move when the bot outputs a command.
3. **Training Dry-Run:** We will create a tiny fake dataset of 1,000 FENs, write the PyTorch Dataloader, and run `train.py` locally to ensure the Loss decreases and Weights & Biases logs the data successfully before deploying to the expensive GCP machine.
