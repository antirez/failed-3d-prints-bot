The goal of this project is to put a Raspberry Pi with a webcam in front of your 3D printer, and receive a message via Telegram when a print failure is detected. This is useful for people that don't print via Octoprint or other systems, but want to be warned if there are issues with what they are printing.

It is composed of two parts:
* A Python program that uses a pre-trained neural network in ONNX format, originating from the [Obico server project](https://github.com/TheSpaghettiDetective/obico-server/tree/release/ml_api/model). This program calls a user-defined command to capture an image, runs the NN to detect failures, stores a `current_score.txt` file with the maximum detection confidence of an error (a float number, from 0 to 1, where numbers > ~0.1 are hints that something is going wrong). The Python script also writes a `processed.png` file where the box with maximum failure confidence is outlined in red.
* A Telegram bot written in C, that periodically checks the `current_score.txt` file, and if it is over a given threshold it sends the image `processed.png` to the configured user / group target.

## Installation

Enter the `bot` directory, compile the bot with `make` and move it into the `detection` library where there is the Python script. They need to stay in the same directory.

Run the bot just with `./mybot`

Run the detection Python program with:

python3 ./run.py --fetch-script image.jpg raspistill -o image.jpg

Where the first argument `image.jpg` is the name of the image that will be produced by the command that follows, and the next arguments are the command to execute to actually produce the image, in this case `raspistill -o image.jpg`, that is good if you are using a Raspberry Pi with the Pi CAM module.

In this mode, the Python program will run forver fetching images and running the neural network again and again. There will be a 5 seconds pause betewen executions.

You can try the Python script with a single example image if you wish:

python3 ./run.py --single test.png

The image `test.png` is included in this software distribution.
The program will create a `processed.png` file with a red box where the failure was detected. The image is also annotated with the current date and time.

## Configuring the Telegram bot

1. Create your bot using the Telegram [@BotFather](https://t.me/botfather).
2. After obtaining your bot API key, store it into a file called `apikey.txt` inside the bot working directory. Alternatively you can use the `--apikey` command line argument to provide your Telegram API key.
4. Run the bot with `./mybot`.
5. Add the bot to your Telegram channel if you want it to say in a group.
6. Talk with the bot with a private message or writing to the group you put the bot into, and set it as the place where you want to receive the failure images by sending the following message: `!target`.
7. Every time you want to receive the current image of your print, just write `!cam`.
8. Once a target for the images was set, it will not accept new `!target` commands, so if you want it to forget the old target and set a new one you need to send a private message to the bot with the `!forget` text from the same user that initially set the target. Then you can use `!target` again.

## Useful information

* Make sure to set the focus of the Raspberry Pi CAM module. You need to rotate the lens. Google for it, it is explained everywhere.
* The Raspberry Pi 3B takes 8 seconds to run the neural network on a single image. That's not too bad. A Macbook M1 takes 80 milliseconds, 100 times less.
* Please **make sure your camera is facing the wall** or alike. Otherwise people could see you and violate your privacy in some way.

## Credits

* The neural network was trained by [Obico](https://github.com/TheSpaghettiDetective/obico-server/).
* The font file used is (Beasted)[https://bjh21.me.uk/bedstead/], a public domain font.
