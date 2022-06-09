# lp-detector
License plate detector based on darknet

For building darknet follow the instructions given [here](https://github.com/AlexeyAB/darknet#requirements-for-windows-linux-and-macos)

Download the model files from [here](https://web.inf.ufpr.br/vri/publications/layout-independent-alpr/) in a sub-directory `data`.

Run the command `pip install -r requirements.txt` to install all dependencies.

Put the image to be detected in the root directory and put its path in the `detect` function in the `main.py` file.

Run `python main.py` to see the result.
