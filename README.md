# Land Cover Classification Data Augmentation

This project involves the augmentation and analysis of land cover classification data. It includes scripts for processing satellite images, visualizing changes in land cover over time, and evaluating model performance. 

## Project Overview

This repository contains scripts for:

- **Data Augmentation:** Enhance land cover classification datasets using various techniques.
- **Inference:** Apply trained models to infer land cover classifications on new data.
- **Comparison and Evolution Analysis:** Compare land cover changes between different years and visualize the evolution of land cover classes.

## Repository Structure

- `README.md`: This file, providing an overview and usage instructions.
- `DATAaug/`: Contains Jupyter notebooks for data augmentation.
- `Inference/`: Scripts for running inference with trained models.
- `Train/`: Training scripts for model training.
- `auxiliary/`: Helper scripts for various auxiliary tasks like evolution analysis and change detection.
- `.idea/`: IDE-specific configuration files (usually not necessary for version control).

## Setup Instructions

### Prerequisites

Ensure you have the following software installed:

- Python 3.x
- Git
- Required Python packages (see `requirements.txt` or specific script documentation for details)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/SierraSergio/LandCoverClassificationDataAugmentation.git
    cd LandCoverClassificationDataAugmentation
    ```

2. Install the necessary Python packages. If a `requirements.txt` file is provided, use:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up any additional environment variables or configurations as needed.

## Usage

### Running Data Augmentation

To perform data augmentation, navigate to the `DATAaug` directory and execute the Jupyter notebook for augmentation:

```bash
cd DATAaug
jupyter notebook AUGmentations_LC.ipynb
