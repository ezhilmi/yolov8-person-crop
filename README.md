# Person Detection and Crop YOLOv8

- Feature Object Tracks
- Person Crop on Unique IDs


## Step to run the code

- Clone the repository

-  Install the requirement
```
pip install -r requirement.txt
```

-  Go to the folder
```
cd person-crop/
```

- Run the code
```
python3 person_crop.py
```

- The output will be in 'cropped' folder

### Using other videos / RTSP

- On line 15 and 16, change the RTSP/Videos path you desired
```
CCTV_RTSP= "rtsp://XXX.XXX.XXX.XXX/channels/XXX"
VIDEPATH= "crowd_ind.mp4"
```
- Line 17, choose either want RTSP (CCTV_RTSP) or Video (VIDEOPATH)

