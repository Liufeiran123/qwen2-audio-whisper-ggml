# qwen2-whisper-ggml
qwen2-audio whisper model ggml inference

qwen2-audio whisper ggml推理基础设施，业余工作记录。

# 生成ggml格式模型
```
models/convert-pt-to-ggml.py ./qwen2-audio-whisper.pt ./github/repos/whisper ./qwen2-audio-whisper-ggml/
```

# 编译代码

```
cmake 
make
```

# 获取whisper输出

```
main
```

感谢
whisper.cpp项目
https://github.com/ggerganov/whisper.cpp
