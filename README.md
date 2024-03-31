# FormAI Dataset Instructions

This repository contains the FormAI dataset, which is compressed to conserve space and facilitate quicker download times. Below you will find instructions on how to decompress the dataset and how to clone this repository.

## Cloning the Repository

To clone the repository and access the files, follow these steps:

1. Open a terminal.
2. Clone the repository using the following command:

    ```
    git clone [Your-Repository-Link-Here]
    ```

    Replace `[Your-Repository-Link-Here]` with the actual link to your repository.

## Decompressing the Dataset

After cloning the repository, you will need to decompress the dataset to use it. The dataset file `FormAI_dataset.csv.gz` is compressed using `gzip`. To decompress it, follow these instructions:

1. Navigate to the directory containing the compressed file. If you just cloned the repository, you can do this with:

    ```
    cd [Your-Repository-Name]/path/to/dataset
    ```

    Adjust the path according to where the dataset is located within your repository.

2. Decompress the file using the following command:

    ```
    gzip -d FormAI_dataset.csv.gz
    ```

    This command will decompress the file and you should see `FormAI_dataset.csv` in the directory.

## Using the Dataset

Once decompressed, you can view the dataset using any CSV-compatible software or tool. For example, to view the first few lines of the CSV in the terminal, you can use:

