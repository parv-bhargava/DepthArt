Install LightGlue
```bash
pip install git+https://github.com/cvg/LightGlue.git
```

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
```

```bash
kaggle models instances versions download oldufo/lightglue/pyTorch/aliked/1
```

```bash
tar -xzf lightglue.tar.gz
```

```bash
cp aliked_lightglue.pth ~/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth
```

```bash
kaggle models instances versions download oldufo/aliked/pyTorch/aliked-n16/1
```

```bash
tar -xzf aliked.tar.gz
```

```bash
cp aliked-n16.pth ~/.cache/torch/hub/checkpoints/aliked-n16.pth
```

```bash
kaggle models instances versions download metaresearch/dinov2/pyTorch/base/1
```
```bash
mkdir -p dinov2/pytorch/base/1
```
```bash
tar -xzf base.tar.gz -C dinov2/pytorch/base/1
```