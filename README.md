# MCS-PCR
Structural TLS Point Cloud Registration via Weighted Hough Transform and Maximal Congruent Subgraph Pairs by Kuo-Liang Chung and Ting-Hsu Chuang

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| [PCL](https://pointclouds.org/) | >= 1.13 | Point cloud I/O, voxel filtering, KD-tree, octree |
| [Eigen3](https://eigen.tuxfamily.org/) | >= 3.4 | Linear algebra |
| [yaml-cpp](https://github.com/jbeder/yaml-cpp) | >= 0.8.0 | Dataset configuration parsing |

A C++17-compatible compiler is required.

## Build

### Docker Build

A Dockerfile is provided that installs all dependencies from source on Ubuntu 22.04:

```bash
git clone https://github.com/ivpml84079/MCS-PCR.git
cd MCS-PCR
docker build -t mcs_pcr .
docker run -it -v "$(PWD):/root/mcs_pcr" -w /root/mcs_pcr --name MCS-PCR mcs_pcr bash
# Inside the container
cmake -S . -B build
cmake --build build -j$(nproc)
```

### Native Build

```bash
git clone https://github.com/ivpml84079/MCS-PCR.git
cd MCS-PCR
cmake -S . -B build
cmake --build build -j$(nproc)
```

This produces two executables:

- `example` -- Register a single pair of point clouds.
- `test_dataset` -- Register all pairs defined in a YAML config file.

## Usage

### Single Pair Registration

```bash
./build/example <source.ply> <target.ply> <ground_truth.txt> <resolution>
```

Example with the provided sample data:

```bash
./build/example datasets/example/source.ply datasets/example/target.ply datasets/example/ground_truth.txt 0.1
```

### Batch Dataset Evaluation

```bash
./build/test_dataset <dataset_name> <resolution> <rot_threshold> <trans_threshold>
```

Example:

```bash
./build/test_dataset Apartment 0.025 5 1
```

Results are saved to `reg_results/<dataset_name>/`:

- `registration_results.txt` -- Per-pair rotation error (deg), translation error (m), and runtime.
- `est_transforms.txt` -- Estimated 4x4 transformation matrices.

### Python Evaluation

A Python evaluation script computes additional metrics (RMSE, MeE) and exports results to Excel:

```bash
python evaluate_result.py \
    --yaml configs/Apartment.yaml \
    --est reg_results/Apartment/est_transforms.txt \
    --reg reg_results/Apartment/registration_results.txt \
    --rot_th 5 --trans_th 1 \
    --out evaluation.xlsx
```

## Datasets

Download the public datasets via the provided script:

```bash
bash datasets/download.sh
```

This downloads and extracts the **Apartment** and **Park** datasets into `datasets/`.

Available dataset configurations (in `configs/`):

| Config | Dataset |
|--------|---------|
| `Apartment.yaml` | Apartment |
| `Boardroom.yaml` | Boardroom |
| `Campus.yaml` | Campus |
| `Park.yaml` | Park |
| `RESSO_6i.yaml` | RESSO figure 6i |
| `RESSO_7a.yaml` | RESSO figure 7a |

Each YAML file specifies the dataset root path, point cloud directory, ground truth directory, and a list of source-target pairs with their ground truth transformation files.

## Project Structure

```
MCS-PCR/
├── include/               # Header files
│   ├── Registration.h     # Main registration class
│   └── ...
├── source/                # Implementation files
├── examples/
│   ├── example.cpp        # Single-pair registration demo
│   └── test_dataset.cpp   # Batch dataset registration
├── configs/               # Dataset YAML configurations
├── datasets/
│   ├── download.sh        # Dataset download script
│   └── example/           # Example point cloud pair with ground truth
├── evaluate_result.py     # Python evaluation script
├── CMakeLists.txt
└── Dockerfile
```
