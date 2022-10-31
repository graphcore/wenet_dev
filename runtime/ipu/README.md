# WeNet runtime on IPU (Intelligence Processing Unit) with Triton Inference Server + IPU backend 

## Background References

about Graphcore: [Graphcore home page](https://www.graphcore.ai)

about IPU: [IPU documents](https://docs.graphcore.ai/en/latest/)

about Triton and Triton for IPU: [Triton](https://github.com/triton-inference-server/server), [Triton For IPU](https://docs.graphcore.ai/projects/poplar-triton-backend/en/latest/introduction.html)


## Usage

step 1: setup environment
before setup the container, please make sure you have setup the ipu partition by checking `env | grep IPUOF_VIPU_API_HOST`

```
bash scripts/setup_image.sh;
bash scripts/setup_container.sh;
```

```
step 2: export model
ckpt: Aishell2, FP16, onnx, [U2++ Conformer](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210618_u2pp_conformer_exp.tar.gz)
```
```
step 3: setup triton server
in the container:
```
bash scripts/setup_offline_server.sh;
```
or
```
bash scripts/setup_online_server.sh;
```
step 4: recognize wavs
```
```
## Precision
```
```
## Performance
```
```
## Performance analyze
```
```


## License


