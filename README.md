# <i>mmDoppler</i> 

<p>
<img align="center"  src="pictures/hardwaresetup.jpg" width="50%"/>
</p>

<hr></hr>

Human activity recognition (HAR) is essential in healthcare, elder care, security, and human-computer interaction, depending on precise sensor data to identify activities. Millimeter wave (mmWave) radar is promising for HAR due to its ability to penetrate non-metallic materials and provide high-resolution sensing. Traditional HAR methods using pointcloud data from sensors like LiDAR or mmWave effectively capture macro-scale activities but struggle with micro-scale activities. This paper introduces <i>mmDoppler</i> , a novel dataset using Commercial off-the-shelf (COTS) mmWave radar, capturing both macro and micro-scale human movements. The dataset includes seven subjects performing 19 distinct activities and employs adaptive doppler resolution to enhance activity recognition. By adjusting the radar's doppler resolution based on the activity type, our system captures subtle movements more precisely. <i>mmDoppler</i>  includes range-doppler heatmaps, offering detailed motion dynamics, with data collected in a controlled environment with single as well as multiple subjects performing activities simultaneously. This dataset bridges the gap in HAR systems, providing a valuable resource for improving the robustness and accuracy of activity recognition using mmWave radar.

## Installation:

To install use the following commands.
```bash
git clone https://github.com/arghasen10/mmDoppler.git
pip install -r requirements.txt
mkdir datasets
cd datasets
```

Please download the dataset through [Google Drive](https://drive.google.com/drive/folders/1fI7C13G-UNubbeyqzopRXs2d2cwGM0F5?usp=sharing) and paste the data in datasets directory.

## Directory Structure


```
mmDoppler
└── models
    └── macro_classifier.py
    └── micro_classifier.py
└── datasets
    └── macro_df_subset.pkl
    └── micro_df_subset.pkl
    └── macro_df.pkl
    └── micro_df.pkl
└── mmwave_demo_visualizer
    └── README.md
```

## Description 

We have provided our dataset in the **datasets** directory. 

To run the activity classifier check **models** directory.

In **mmwave_demo_visualizer** directory we have provided the instructions to run the demo visualizer for data collection as well as real time data visualization. This implementation is made by Texas Instruments and we have slightly modified the version to enable data collection and data annotations.

In **suppllementary_plots** directory we have provided the source code for the plots we added in the paper. 

<hr>

## Dataset Importance

| Datasets  | Modality                                                 | Activity Type              | \# Classes | Granularity     | \# Frames | Effective Duration (s) | Multi-Subjects |
| --------- | -------------------------------------------------------- | -------------------------- | ---------- | --------------- | --------- | ---------------------- | -------------- |
| mRI       | mmWave pointcloud, RGB, Depth Camera and IMU signals     | Pose estimation            | 12         | Macro           | 160k      | 5333                   | No             |
| mm-FI     | RGB, Depth Camera, LiDAR and mmWave pointcloud, WiFi CSI | Daily activities           | 27         | Macro           | 320k      | 10666                  | No             |
| MiliPoint | mmWave pointcloud                                        | Daily activities           | 49         | Macro           | 545k      | 18166                  | No             |
| RadHAR    | mmWave pointcloud                                        | Exercise                   | 5          | Macro           | 167k      | 5566                   | No             |
| <b><i>mmDoppler</i></b> | <b>mmWave pointcloud, Range-Doppler heatmaps</b>                | <b>Daily activities, Exercise</b> | <b>19</b>         | <b>Macro and Micro</b> | <b>75k</b>       | <b>23100</b>                  | <b>Yes</b>            |

## Reference
To refer <i>mmDoppler</i> dataset, please cite the following work.

BibTex Reference:
```
@article{sen2024mmdoppler,
  title={Human Activity Dataset Using Adaptive Doppler Resolution with mmWave Radar},
  author={Sen, Argha and Das, Anirban and Pradhan, Swadhin and Chakraborty, Sandip},
  year={2024}
}
For questions and general feedback, contact Argha Sen (arghasen10@gmail.com).

