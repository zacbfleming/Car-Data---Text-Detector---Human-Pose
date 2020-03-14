# Car-Data---Text-Detector---Human-Pose
3 neural network models - Human pose estimation, car metadata and text detector

3 models in one application. 
Car info - CAR_META -a vehicle attribute classification algorithm. The model is able to classify car type and color.

Text detector - TEXT - The text detector model is based on PixelLink architecture with MobileNetV2-like as a backbone for indoor/outdoor scenes. The network outputs 2 blobs

a) [1x2x192x320] - logits related to text/no-text classification for each pixel

b) [1x16x192x320] - logits related to linkage between pixels and their neighbors.

Human pose estimation - POSE - is a multi person pose estimation network that uses the OpenPose approach. The model uses a tuned MobileNet v1 for feature extraction for every person in the image. The pose may contain up to 18 keypoints - ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees and ankles. The model outputs 2 blobs. The first blob contains keypoint pairwise relations (part affinity fields), while the second blob contains keypoint heatmaps. Currently the app outputs heatmaps for a given image.


Installation / Configuration

opencv 4.0.1
numpy 1.18.0
openvino 2020.1

Copy all the files in the manifest to the same folder and run...

python3 app.py -m -i -t

The app will write an image according to the choice of type in the same ffolder. 

switches

App with Inference Engine [-h] -i I -m M -t T [-c C] [-d D]

required arguments:
  -i I  The location of the input image
  -m M  The location of the model XML file
  -t T  The type of model: POSE, TEXT or CAR_META

optional arguments:
  -c C  CPU extension file location, if applicable
  -d D  Device, if not CPU (GPU, FPGA, MYRIAD)
  
 ( i.e. python3 app.py -i blue.jpg -t CAR_META -m vehicle-attributes-recognition-barrier-0039.xml)
  
  
