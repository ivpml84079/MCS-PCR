# MCS-PCR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Structural TLS Point Cloud Registration via Weighted Hough Transform and Maximal Congruent Subgraph Pairs by Kuo-Liang Chung, Ting-Hsu Chuang, and Hao-En Huang.

## Dependencies

```
PCL (>= 1.13.0)
Eigen3
yaml-cpp (0.8.0)
```

## Build

Open `TopologicalConsistencyRegis.sln` in **Visual Studio 2022** and build the `Release | x64` configuration.

Ensure the following paths are set correctly in [`PCL1.13.0_release_x64.props`](PCL1.13.0_release_x64.props) and [`yaml0.8.0_release_x64`](yaml0.8.0_release_x64) property sheets:

- PCL 1.13.0 include and library directories
- yaml-cpp 0.8.0 include and library directories

## Usage

```bash
TopologicalConsistencyRegis.exe <dataset_name> <resolution> <max_line_num> <angle_tolerance> <fac_epsilon> <fac_tau>
```

### Parameters

| Parameter | Description | Example |
|---|---|---|
| `<dataset_name>` | Name of the dataset (must match a YAML config in `configs/`) | `Park` |
| `<resolution>` | Voxel resolution in meters | `0.1` |
| `<max_line_num>` | Maximum number of lines to detect per cloud | `20` |
| `<angle_tolerance>` | Tolerance for matching line angles (degrees) | `3.0` |
| `<fac_epsilon>` | Distance tolerance factor (multiplied by resolution) | `2.0` |
| `<fac_tau>` | Z-estimation voxel size factor | `2.0` |

### Example

```bash
TopologicalConsistencyRegis.exe Park 0.1 20 3.0 2.0 2.0
```

## Configuration

Each dataset requires a YAML configuration file under `configs/<dataset_name>.yaml`:

```yaml
dataset_name: "Park"
root: "E:\\Dataset\\Park\\"
groundtruth: "3-GroundTruth\\"
raw_data: "1-RawPointCloud\\"
pairs:
  - source: "1.ply"
    target: "2.ply"
    transformation_file: "1-2\\transformation.txt"
```

Supported datasets include: `Apartment`, `Boardroom`, `Campus`, `Park`, `RESSO_6i`, `RESSO_7a`.


## Testing Environment

* Windows 11 Pro
* Visual Studio 2022 (MSVC v143)
* ISO C++17
* x64

## Contact

If you have any questions, please email us via  
Kuo-Liang Chung: [klchung01@gmail.com](mailto:klchung01@gmail.com)  
Ting-Hsu Chuang: [0820timothy@gmail.com](mailto:0820timothy@gmail.com)  
