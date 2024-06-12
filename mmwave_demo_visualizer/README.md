# mmWave-Demo-Visualizer_2.1.0
TI Demo Visualizer for IWR1642EVM, AWR1642EVM

## Installation Guide

To install mmWave Demo Visualizer from Texas Instruments first go to this [link](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/) and select SDK [2.1.0](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/2.1.0/).

Now go to Help and select Download or Clone Visualizer. FInally you need to download and install the entire repository in your machine.

Now clone this repository with the following command.
```bash
git clone https://github.com/arghasen10/mmWave-Demo-Visualizer.git

```
Once you clone the repository copy all the content and paste it in the installaed mmWave-demo-visualizer directory i.e. **C:\Users\UserName\guicomposer\runtime\gcruntime.v11\mmWave_Demo_Visualizer**

Once you are done with the installation run 
```bash
launcher.exe
```
Finally using this tool you can save mmWave data in your local machine. Data will be saved in a txt file in JSON format.

## Process Collected datasets

To process the collected raw datsets in `.txt` format, first keep all the files inside `datasets/raw_datasets` directory. Also keep the provided script `process_data.py` in the same directory `datasets/raw_datasets`. It will process the dataset and will store it in `.pkl` format as pandas dataframe.