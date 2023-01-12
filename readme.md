# Diff tool for OD inference (DiffOD)

Streamlit App for compare difference for 2 different Object detection model  


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
