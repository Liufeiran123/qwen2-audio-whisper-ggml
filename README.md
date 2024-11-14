# qwen2-audio-whisper-ggml
qwen2-audio whisper model ggml inference

qwen2-audio whisper ggml推理基础设施，业余工作记录。

## clone代码
```
git clone https://github.com/Liufeiran123/qwen2-audio-whisper-ggml.git
```

## 切换到根目录
```
cd qwen2-audio-whisper-ggml
```

## 生成ggml格式模型
```
python ./models/convert-pt-to-ggml.py ./qwen2-audio-whisper.pt ./github/repos/whisper ./qwen2-audio-whisper-ggml/
```

## 编译代码

```
cmake -DGGML_CUDA=ON -DCMAKE_VERBOSE_MAKEFILE=ON -DWHISPER_BUILD_EXAMPLES=ON .
make
```

## 获取whisper输出
```
./bin/main -f ./samples/jfk.wav
```

感谢
whisper.cpp项目
https://github.com/ggerganov/whisper.cpp
