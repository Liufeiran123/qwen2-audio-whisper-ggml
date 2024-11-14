# qwen2-whisper-ggml
qwen2-audio whisper model ggml inference

qwen2-audio whisper ggml推理基础设施

Quick start
First clone the repository:

git clone https://github.com/ggerganov/whisper.cpp.git
Navigate into the directory:

cd whisper.cpp
Then, download one of the Whisper models converted in ggml format. For example:

sh ./models/download-ggml-model.sh base.en
Now build the main example and transcribe an audio file like this:

# build the main example
make -j

# transcribe an audio file
./main -f samples/jfk.wav

cmake -DGGML_CUDA=ON -DCMAKE_VERBOSE_MAKEFILE=ON -DWHISPER_BUILD_EXAMPLES=ON .

make
