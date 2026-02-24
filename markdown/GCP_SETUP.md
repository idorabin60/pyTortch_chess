# GCP Setup & Training Guide 🚀

This is the exact playbook to use your $300 GCP credit to train the `Antigravity Chess AI` to Grandmaster levels.

## 1. Spin up the Virtual Machine

Head to the **Google Cloud Console > Compute Engine > VM Instances > Create Instance**.

Use the following exact specifications:
*   **Name:** `chess-training-vm-1`
*   **Region:** Us-central1 (or any zone with L4 availability)
*   **Machine configuration:** `GPU`
*   **GPU Type:** `NVIDIA L4` (Keep the count at `1` to start)
*   **Machine type:** `g2-standard-4` (4 vCPUs, 16 GB memory) or `g2-standard-8` (8 vCPUs, 32 GB memory)
*   **Boot Disk OS:** Deep Learning on Linux (Ubuntu 22.04 with PyTorch and CUDA 12 pre-installed)
*   **Disk Size:** 200 GB SSD (You need space to download the massive datasets!)

**Cost Analysis:** An NVIDIA L4 generally costs ~`$0.60/hr`. Training for a full week (168 hours) will cost approximately `$100`, well within your budget!

## 2. Upload the Code

When the machine is running, simply click the **"SSH"** button in the GCP console to open a terminal in your browser.

You need to securely upload 4 files from your MacBook to the new VM:
1.  `network.py` (The brain architecture)
2.  `data_processing.py` (The math tensor conversions)
3.  `dataset_loader.py` (The PyTorch batching logic)
4.  `train_gcp.py` (The training loop)

*(You can use the native "Upload File" button in the GCP SSH window, or use `scp` from your macbook terminal!)*

## 3. Prepare the Environment

In the GCP SSH terminal, install the dependencies, including `wandb` for our remote logging dashboard!

```bash
pip install torch numpy chess wandb
```

Login to Weights & Biases so `train_gcp.py` knows where to send the logs:
```bash
wandb login
```
(It will ask you to paste an API key which you can get for free at wandb.ai)

## 4. Download The Golden Dataset

We are using the `r2dev2` 16-million position dataset. You can download it directly onto the VM.
```bash
# This is an example wget command if you find a direct link, 
# or you can use the Kaggle CLI to pull it directly onto the VM!
pip install kaggle
kaggle datasets download -d [dataset-id]
```

*Note: You will need to extract the dataset, and slightly modify `train_gcp.py` to point to the actual path of the CSV/JSON file instead of the "fake_fens" list in the code!*

## 5. Begin Training!

To train the math weights, you can't just run `python3 train_gcp.py` directly, because if you close your laptop or the SSH window disconnects, the training will instantly stop!

We use a tool called `tmux` (terminal multiplexer) to keep it running forever in the background:
```bash
tmux new -s chess_training
python3 train_gcp.py
```

Now you can close your laptop. The L4 GPU will begin churning through the 16 million positions!

## 6. Remote Logging (WandB Dashboard)

Open [wandb.ai](https://wandb.ai) on your MacBook or phone.
Because of the `wandb.log()` line we added to `train_gcp.py`, you will see a gorgeous dashboard automatically appear!

**What to watch for:**
*   **Loss Curve:** Look at the chart for "Loss". It should look like a steep ski slope going down, eventually leveling out near 0 over the 48 hours. If it goes up, or zigzags wildly, something is wrong!
*   **GPU Utilization:** Ensure it sits around `90-100%`. If it's at `20%`, the CPU `DataLoader` is struggling to feed the FENs fast enough, and you should increase the `num_workers` in `train_gcp.py`!

## 7. The End Game

When the 100 Epochs finish in a few days, the script will automatically create a file called `best_model.pth`.
SSH back in, download that single file to your MacBook, place it in the same directory as `uci.py`, uncomment the `.load_state_dict()` line in `integration.py`, and link it to Arena!

Your Zero-to-Hero AI is complete!
