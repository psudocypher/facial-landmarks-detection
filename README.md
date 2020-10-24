# Facial-Landmarks-Detection
> ## Facial landmarks detection using dlib and opencv
> ### You can download the shape_predictor_68_face_landmarks.dat by clicking [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Extract the content of zip file in the same folder where you have saved your source code.


<img src = "facial landmarks image/landmarks.png" width = 300>   ![landmark GIF](GIF/ezgif.com-gif-maker.gif)

download the reprository and run "face landmark detection.py"
```python
python "face landmark detection.py"
```

if you don't have dlib and cv2 packages on your computer, install them by :
```
pip install dlib
```
and
```
pip install opencv-python 
```
if the above approch doesn't work for you, then try creating enviorment first and then follow the similar approch.
## Future work
> We can use the landmarks predicted by the above program as follow:
>> 1. We can use eye landmaks to detect drowsiness.
>> 2. We can also made a snapchat filter using these predicted landmarks and many more.
