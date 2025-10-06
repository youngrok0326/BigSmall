# Inter-Scaling

## General Information

Our implementation is based on [Unsloth](https://github.com/unslothai/unsloth).

## Reproducing the Experiments

- To install relevant dependencies, install `uv` and enter

  ``` bash
  uv sync
  ```

- To re-run the single-GPU experiments, edit `config/train.yaml`, and enter

  ``` bash
  mkdir -p checkpoints
  uv run python3 train.py
  ```

- To evaluate the saved checkpoints of a single-GPU experiment run, edit `config/test.yaml`, and enter

  ``` bash
  uv run python3 evaluate-run.py
  ```

- To evaluate the saved checkpoints combined with *Inference Scaling* of a single-GPU experiment run, edit `config/decode_run_eval.yaml`, and enter

  ``` bash
  uv run python3 evaluate-run-decode.py
  ```

- To evaluate the SMC with a base model run, edit `config/decode_eval.yaml`, and enter

  ``` bash
  uv run python3 evaluate-decode.py
  ```