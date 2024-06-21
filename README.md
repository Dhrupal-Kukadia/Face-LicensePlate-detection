
<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">Face and License Plate Detector</h3>

  <p align="center">
    The repository contains a script and related modules for detecting faces and license plates in frames of a driving sequence.
    <br />
    ·
    <a href="https://github.com/ankitoscar/face-lp-detector/issues">Report Bug</a>
    ·
    <a href="https://github.com/ankitoscar/face-lp-detector/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The project aims at blurring the face and license plates captured in the frames of a driving sequence. The dataset used is similar to the [India Driving Dataset](https://idd.insaan.iiit.ac.in/).

The detection task divided into face and license plate detection. Face detection is done using [Retina-Face](https://pypi.org/project/retina-face/), while license plate detection is done using [this approach](https://web.inf.ufpr.br/vri/publications/layout-independent-alpr/).

### RetinaFace

The algorithm used for face detection is based on this [paper](https://arxiv.org/pdf/1905.00641). RetinaFace is a robust single-stage face detection model designed for high-precision face localization in images. Its architecture utilizes a single deep neural network with a multi-task loss function, incorporating three key tasks: face detection bounding box regression, facial landmark localization, and detection of face rotation angles. RetinaFace achieves state-of-the-art performance by leveraging a lightweight and efficient design suitable for real-time applications.

![pic1.png]()

![pic2.png]()

### License Plate Detection

The algorithm used for license plate detection is based on this [paper](https://arxiv.org/pdf/1909.01754.pdf). It states that in order to detect license plate, we need to first detect vehicles, which will be followed by cropping the vehicle images which are used for detecting license plates. The task is carried out by two YOLO-based networks, YOLO-v2 for vehicle detection and Fast YOLO-v2.

![pic1.png](https://github.com/ankitoscar/face-lp-detector/blob/main/images/pic1.png)

![pic2.png](https://github.com/ankitoscar/face-lp-detector/blob/main/images/pic2.png)

<p align="center">License Plate Detection</p>

### Blurring

Blurring is done by make two instances of the frame, one is used for getting detections and another is used for blur the bounding box and saving the image.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

* [Python](https://www.python.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [RetinaFace](https://pypi.org/project/retina-face/)
* [Darknet](https://github.com/AlexeyAB/darknet)
* [OpenCV](https://opencv.org/)
* [Pillow](https://pillow.readthedocs.io/)
* [Matplotlib](https://matplotlib.org/)
* [Numpy](https://numpy.org/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

This code was run on a system with the following specifications:

* Windows 11(64 bit)
* 8GB RAM
* NVIDIA GeForce GTX 1650(4 GB), with CUDA(ver. 11.2) and CuDNN(ver. 8.0)
* Python 3.9.7

### Installation

1. Clone the repo.
   ```sh
   git clone https://github.com/ankitoscar/face-lp-detector.git
   ```
2. Create a virtual environment using *virtualenv* and activate it.
   ```sh
    virtualenv .\{your-environment-name}
    .\{your-environment-name}\Scripts\activate
   ```
3. Install all the required packages from *requirements.txt*. 
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Running the main script
To run the main script `run.py`, follow these steps:

* Unzip the dataset in a folder named *IDD2_Subset* in the root directory of the repository.

* Run the following command in the shell.
```sh
  python run.py
```

* The script will start running.

Since the code involves read, write, object detection and image processing, it is computationally expensive and can take around an hour to finish.

### Doing ablation study

Follow these steps for ablation study:

* Copy the dataset folder in the root directory itself and rename it as `ablation-study`.

* Run the following command in the shell to delete some files in order to perform the study on a subset of the dataset.
```sh
  python script.py "delete"
``` 

* Then, modify the thresholds of face and license plates. For license plates, go to `detect.py` and change `VEHICLE_DETECTOR_THRESHOLD` and `LP_DETECTOR_THRESHOLD`.

* Write your thresholds in `ablation-study.txt` as **study-number**.

* Run the following command in the shell to start ablation study.
```sh
  python script.py "study-number"
```

* This will create a seperate directory in the `ablation-study` directory for **study-number**.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Project Link: [https://github.com/ankitoscar/face-lp-detector](https://github.com/ankitoscar/face-lp-detector)

<p align="right">(<a href="#top">back to top</a>)</p>
