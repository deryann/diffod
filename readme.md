# Diff (DiffOD)

Streamlit App to show difference for 2 different Object Detection models  
![](figure/demo-v001.png)

## Docker command reference 
### How to build it and run it.

```bash 
# build it
docker build -t diffod:latest .

# run it (CPU)
docker run -it -p 127.0.0.1:8501:8501 diffod:latest

# run it (GPU)
docker run -it 
```

### Use docker compose 
to run app (include yolov5 and yolov8 interface) and yolov7
```bash
docker compose up
```

## Contact
For DiffOd bugs and feature requests please visit [GitHub Issues](https://github.com/deryann/diffod/issues).
For professional support please [Contact Us](mailto:deryann@gmail.com).


## Reference 
- [yolov5](https://github.com/ultralytics/yolov5)
- [yolov8](https://github.com/ultralytics/ultralytics)
