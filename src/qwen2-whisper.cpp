#include "qwen2-whisper.h"

#ifdef WHISPER_USE_COREML
#include "coreml/whisper-encoder.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#ifdef WHISPER_USE_OPENVINO
#include "openvino/whisper-openvino-encoder.h"
#endif

#ifdef GGML_USE_CANN
#include "ggml-cann.h"
#endif

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <atomic>
#include <algorithm>
#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include <random>
#include <functional>
#include <codecvt>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if defined(GGML_BIG_ENDIAN)
#include <bit>

template<typename T>
static T byteswap(T value) {
    return std::byteswap(value);
}

template<>
float byteswap(float value) {
    return std::bit_cast<float>(byteswap(std::bit_cast<std::uint32_t>(value)));
}

template<typename T>
static void byteswap_tensor_data(ggml_tensor * tensor) {
    T * datum = reinterpret_cast<T *>(tensor->data);
    for (int i = 0; i < ggml_nelements(tensor); i++) {
        datum[i] = byteswap(datum[i]);
    }
}

static void byteswap_tensor(ggml_tensor * tensor) {
    switch (tensor->type) {
        case GGML_TYPE_I16: {
            byteswap_tensor_data<int16_t>(tensor);
            break;
        }
        case GGML_TYPE_F16: {
            byteswap_tensor_data<ggml_fp16_t>(tensor);
            break;
        }
        case GGML_TYPE_I32: {
            byteswap_tensor_data<int32_t>(tensor);
            break;
        }
        case GGML_TYPE_F32: {
            byteswap_tensor_data<float>(tensor);
            break;
        }
        default: { // GML_TYPE_I8
            break;
        }
    }
}

#define BYTESWAP_VALUE(d) d = byteswap(d)
#define BYTESWAP_FILTERS(f)            \
    do {                              \
        for (auto & datum : f.data) { \
            datum = byteswap(datum);  \
        }                             \
    } while (0)
#define BYTESWAP_TENSOR(t)       \
    do {                         \
        byteswap_tensor(t); \
    } while (0)
#else
#define BYTESWAP_VALUE(d) do {} while (0)
#define BYTESWAP_FILTERS(f) do {} while (0)
#define BYTESWAP_TENSOR(t) do {} while (0)
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define WHISPER_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define WHISPER_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define WHISPER_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

WHISPER_ATTRIBUTE_FORMAT(2, 3)
static void whisper_log_internal        (ggml_log_level level, const char * format, ...);
static void whisper_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define WHISPER_LOG_ERROR(...) whisper_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define WHISPER_LOG_WARN(...)  whisper_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define WHISPER_LOG_INFO(...)  whisper_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)

// define this to enable verbose trace logging - useful for debugging purposes
//#define WHISPER_DEBUG

#if defined(WHISPER_DEBUG)
#define WHISPER_LOG_DEBUG(...) whisper_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#define WHISPER_LOG_DEBUG(...)
#endif

#define WHISPER_ASSERT(x) \
    do { \
        if (!(x)) { \
            WHISPER_LOG_ERROR("WHISPER_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#define WHISPER_MAX_DECODERS 8
#define WHISPER_MAX_NODES 4096

//
// ggml helpers
//

static bool ggml_graph_compute_helper(
          struct ggml_cgraph * graph,
        std::vector<uint8_t> & buf,
                         int   n_threads,
         ggml_abort_callback   abort_callback,
                        void * abort_callback_data) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);

    plan.abort_callback      = abort_callback;
    plan.abort_callback_data = abort_callback_data;

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    return ggml_graph_compute(graph, &plan);
}

static bool ggml_graph_compute_helper(
      ggml_backend_sched_t   sched,
        struct ggml_cgraph * graph,
                       int   n_threads) {

    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }
#ifdef GGML_USE_BLAS
        if (ggml_backend_is_blas(backend)) {
            ggml_backend_blas_set_n_threads(backend, n_threads);
        }
#endif
    }

    bool t = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_reset(sched);
    return t;
}

// faster matrix multiplications for tensors that do not have dimension 0 divisible by "pad"
// the idea is to represent the original matrix multiplication:
//
//   Z = X @ Y
//
// with the sum of two matrix multiplications:
//
//   Z = (X_0 @ Y_0) + (X_1 @ Y_1)
//
// here X_0 and Y_0 are views of X and Y that have dimension 0 divisible by "pad"
// and X_1 and Y_1 are the remaining views. X_1 and Y_1 end up being small matrices that can be processed with more
// general-purpose kernels
//
static struct ggml_tensor * ggml_mul_mat_pad(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y, int pad = 32) {
    // use padding only if dimension 0 is at least 8 times larger than the padding
    // else we won't get much benefit from the optimization
    const int n_pad_req = 8;

    if (x->ne[0] % pad == 0 || x->ne[0] / pad < n_pad_req) {
        return ggml_mul_mat(ctx, x, y);
    }

    struct ggml_tensor * x_0 = ggml_view_3d(ctx, x, (x->ne[0]/pad)*pad, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    struct ggml_tensor * x_1 = ggml_view_3d(ctx, x,  x->ne[0]%pad,      x->ne[1], x->ne[2], x->nb[1], x->nb[2], x_0->ne[0]*x_0->nb[0]);

    struct ggml_tensor * y_0 = ggml_view_3d(ctx, y, (y->ne[0]/pad)*pad, y->ne[1], y->ne[2], y->nb[1], y->nb[2], 0);
    struct ggml_tensor * y_1 = ggml_view_3d(ctx, y,  y->ne[0]%pad,      y->ne[1], y->ne[2], y->nb[1], y->nb[2], y_0->ne[0]*y_0->nb[0]);

    return ggml_add(ctx,
            ggml_mul_mat(ctx, x_0, y_0),
            ggml_mul_mat(ctx, x_1, y_1));
}

// TODO: check if other platforms can benefit from this optimization
// TODO: CUDA is currently broken - seems ggml_mul_mat does not handle views correctly
#if defined(GGML_USE_METAL)
#define ggml_mul_mat ggml_mul_mat_pad
#endif

// available whisper models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_TINY,
    MODEL_BASE,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
};

static const std::map<e_model, std::string> g_model_name = {
    { MODEL_UNKNOWN,  "unknown"  },
    { MODEL_TINY,     "tiny"     },
    { MODEL_BASE,     "base"     },
    { MODEL_SMALL,    "small"    },
    { MODEL_MEDIUM,   "medium"   },
    { MODEL_LARGE,    "large"    },
};

static const std::map<std::string, std::pair<int, std::string>> g_lang = {
    { "en",  { 0,  "english",         } },
    { "zh",  { 1,  "chinese",         } },
    { "de",  { 2,  "german",          } },
    { "es",  { 3,  "spanish",         } },
    { "ru",  { 4,  "russian",         } },
    { "ko",  { 5,  "korean",          } },
    { "fr",  { 6,  "french",          } },
    { "ja",  { 7,  "japanese",        } },
    { "pt",  { 8,  "portuguese",      } },
    { "tr",  { 9,  "turkish",         } },
    { "pl",  { 10, "polish",          } },
    { "ca",  { 11,  "catalan",        } },
    { "nl",  { 12,  "dutch",          } },
    { "ar",  { 13,  "arabic",         } },
    { "sv",  { 14,  "swedish",        } },
    { "it",  { 15,  "italian",        } },
    { "id",  { 16,  "indonesian",     } },
    { "hi",  { 17,  "hindi",          } },
    { "fi",  { 18,  "finnish",        } },
    { "vi",  { 19,  "vietnamese",     } },
    { "he",  { 20,  "hebrew",         } },
    { "uk",  { 21,  "ukrainian",      } },
    { "el",  { 22,  "greek",          } },
    { "ms",  { 23,  "malay",          } },
    { "cs",  { 24,  "czech",          } },
    { "ro",  { 25,  "romanian",       } },
    { "da",  { 26,  "danish",         } },
    { "hu",  { 27,  "hungarian",      } },
    { "ta",  { 28,  "tamil",          } },
    { "no",  { 29,  "norwegian",      } },
    { "th",  { 30,  "thai",           } },
    { "ur",  { 31,  "urdu",           } },
    { "hr",  { 32,  "croatian",       } },
    { "bg",  { 33,  "bulgarian",      } },
    { "lt",  { 34,  "lithuanian",     } },
    { "la",  { 35,  "latin",          } },
    { "mi",  { 36,  "maori",          } },
    { "ml",  { 37,  "malayalam",      } },
    { "cy",  { 38,  "welsh",          } },
    { "sk",  { 39,  "slovak",         } },
    { "te",  { 40,  "telugu",         } },
    { "fa",  { 41,  "persian",        } },
    { "lv",  { 42,  "latvian",        } },
    { "bn",  { 43,  "bengali",        } },
    { "sr",  { 44,  "serbian",        } },
    { "az",  { 45,  "azerbaijani",    } },
    { "sl",  { 46,  "slovenian",      } },
    { "kn",  { 47,  "kannada",        } },
    { "et",  { 48,  "estonian",       } },
    { "mk",  { 49,  "macedonian",     } },
    { "br",  { 50,  "breton",         } },
    { "eu",  { 51,  "basque",         } },
    { "is",  { 52,  "icelandic",      } },
    { "hy",  { 53,  "armenian",       } },
    { "ne",  { 54,  "nepali",         } },
    { "mn",  { 55,  "mongolian",      } },
    { "bs",  { 56,  "bosnian",        } },
    { "kk",  { 57,  "kazakh",         } },
    { "sq",  { 58,  "albanian",       } },
    { "sw",  { 59,  "swahili",        } },
    { "gl",  { 60,  "galician",       } },
    { "mr",  { 61,  "marathi",        } },
    { "pa",  { 62,  "punjabi",        } },
    { "si",  { 63,  "sinhala",        } },
    { "km",  { 64,  "khmer",          } },
    { "sn",  { 65,  "shona",          } },
    { "yo",  { 66,  "yoruba",         } },
    { "so",  { 67,  "somali",         } },
    { "af",  { 68,  "afrikaans",      } },
    { "oc",  { 69,  "occitan",        } },
    { "ka",  { 70,  "georgian",       } },
    { "be",  { 71,  "belarusian",     } },
    { "tg",  { 72,  "tajik",          } },
    { "sd",  { 73,  "sindhi",         } },
    { "gu",  { 74,  "gujarati",       } },
    { "am",  { 75,  "amharic",        } },
    { "yi",  { 76,  "yiddish",        } },
    { "lo",  { 77,  "lao",            } },
    { "uz",  { 78,  "uzbek",          } },
    { "fo",  { 79,  "faroese",        } },
    { "ht",  { 80,  "haitian creole", } },
    { "ps",  { 81,  "pashto",         } },
    { "tk",  { 82,  "turkmen",        } },
    { "nn",  { 83,  "nynorsk",        } },
    { "mt",  { 84,  "maltese",        } },
    { "sa",  { 85,  "sanskrit",       } },
    { "lb",  { 86,  "luxembourgish",  } },
    { "my",  { 87,  "myanmar",        } },
    { "bo",  { 88,  "tibetan",        } },
    { "tl",  { 89,  "tagalog",        } },
    { "mg",  { 90,  "malagasy",       } },
    { "as",  { 91,  "assamese",       } },
    { "tt",  { 92,  "tatar",          } },
    { "haw", { 93,  "hawaiian",       } },
    { "ln",  { 94,  "lingala",        } },
    { "ha",  { 95,  "hausa",          } },
    { "ba",  { 96,  "bashkir",        } },
    { "jw",  { 97,  "javanese",       } },
    { "su",  { 98,  "sundanese",      } },
    { "yue", { 99,  "cantonese",      } },
};

// [EXPERIMENTAL] Token-level timestamps with DTW
static const whisper_ahead g_aheads_tiny_en[]   = { {1, 0}, {2, 0}, {2, 5}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4} };
static const whisper_ahead g_aheads_tiny[]      = { {2, 2}, {3, 0}, {3, 2}, {3, 3}, {3, 4}, {3, 5} };
static const whisper_ahead g_aheads_base_en[]   = { {3, 3}, {4, 7}, {5, 1}, {5, 5}, {5, 7} };
static const whisper_ahead g_aheads_base[]      = { {3, 1}, {4, 2}, {4, 3}, {4, 7}, {5, 1}, {5, 2}, {5, 4}, {5, 6} };
static const whisper_ahead g_aheads_small_en[]  = { {6, 6}, {7, 0}, {7, 3}, {7, 8}, {8, 2}, {8, 5}, {8, 7}, {9, 0}, {9, 4}, {9, 8}, {9, 10}, {10, 0}, {10, 1}, {10, 2}, {10, 3}, {10, 6}, {10, 11}, {11, 2}, {11, 4} };
static const whisper_ahead g_aheads_small[]     = { {5, 3}, {5, 9}, {8, 0}, {8, 4}, {8, 7}, {8, 8}, {9, 0}, {9, 7}, {9, 9}, {10, 5} };
static const whisper_ahead g_aheads_medium_en[] = { {11, 4}, {14, 1}, {14, 12}, {14, 14}, {15, 4}, {16, 0}, {16, 4}, {16, 9}, {17, 12}, {17, 14}, {18, 7}, {18, 10}, {18, 15}, {20, 0}, {20, 3}, {20, 9}, {20, 14}, {21, 12} };
static const whisper_ahead g_aheads_medium[]    = { {13, 15}, {15, 4}, {15, 15}, {16, 1}, {20, 0}, {23, 4} };
static const whisper_ahead g_aheads_large_v1[]  = { {9, 19}, {11, 2}, {11, 4}, {11, 17}, {22, 7}, {22, 11}, {22, 17}, {23, 2}, {23, 15} };
static const whisper_ahead g_aheads_large_v2[]  = { {10, 12}, {13, 17}, {16, 11}, {16, 12}, {16, 13}, {17, 15}, {17, 16}, {18, 4}, {18, 11}, {18, 19}, {19, 11}, {21, 2}, {21, 3}, {22, 3}, {22, 9}, {22, 12}, {23, 5}, {23, 7}, {23, 13}, {25, 5}, {26, 1}, {26, 12}, {27, 15} };
static const whisper_ahead g_aheads_large_v3[]  = { {7, 0}, {10, 17}, {12, 18}, {13, 12}, {16, 1}, {17, 14}, {19, 11}, {21, 4}, {24, 1}, {25, 6} };
static const whisper_ahead g_aheads_large_v3_turbo[]  = { {2, 4}, {2, 11}, {3, 3}, {3, 6}, {3, 11}, {3, 14} };

static const std::map<whisper_alignment_heads_preset, whisper_aheads> g_aheads {
    { WHISPER_AHEADS_TINY_EN,   {  8, g_aheads_tiny_en   } },
    { WHISPER_AHEADS_TINY,      {  6, g_aheads_tiny      } },
    { WHISPER_AHEADS_BASE_EN,   {  5, g_aheads_base_en   } },
    { WHISPER_AHEADS_BASE,      {  8, g_aheads_base      } },
    { WHISPER_AHEADS_SMALL_EN,  { 19, g_aheads_small_en  } },
    { WHISPER_AHEADS_SMALL,     { 10, g_aheads_small     } },
    { WHISPER_AHEADS_MEDIUM_EN, { 18, g_aheads_medium_en } },
    { WHISPER_AHEADS_MEDIUM,    {  6, g_aheads_medium    } },
    { WHISPER_AHEADS_LARGE_V1,  {  9, g_aheads_large_v1  } },
    { WHISPER_AHEADS_LARGE_V2,  { 23, g_aheads_large_v2  } },
    { WHISPER_AHEADS_LARGE_V3,  { 10, g_aheads_large_v3  } },
    { WHISPER_AHEADS_LARGE_V3_TURBO, { 6, g_aheads_large_v3_turbo } },
};

static std::vector<uint32_t> get_alignment_heads_by_layer(const whisper_context_params & cparams, int il, int32_t n_text_layer, int32_t n_head);

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

struct whisper_vocab {
    using id    = int32_t;
    using token = std::string;

    int n_vocab = 51864;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    // reference: https://github.com/openai/whisper/blob/248b6cb124225dd263bb9bd32d060b6517e067f8/whisper/tokenizer.py#L334-L349
    id token_eot        = 50256;
    id token_sot        = 50257;
    // task tokens (used only for multilingual models)
    id token_translate  = 50357;
    id token_transcribe = 50358;
    // other special tokens
    id token_solm       = 50359; // [TDRZ] used by tinydiarize models to indicate speaker turn
    id token_prev       = 50360;
    id token_nosp       = 50361;
    id token_not        = 50362; // no timestamps
    id token_beg        = 50363; // begin timestamps

    bool is_multilingual() const {
        return n_vocab >= 51865;
    }

    int num_languages() const {
        return n_vocab - 51765 - (is_multilingual() ? 1 : 0);
    }
};

struct whisper_batch {
    int32_t n_tokens;

    whisper_token  *  token;
    whisper_pos    *  pos;
    int32_t        *  n_seq_id; // always 1, here for consistency with llama.cpp
    whisper_seq_id ** seq_id;   // null terminated
    int8_t         *  logits;
};

static struct whisper_batch whisper_batch_init(int32_t n_tokens, int32_t n_seq_max) {
    whisper_batch batch = { 0, nullptr, nullptr, nullptr, nullptr, nullptr, };

    batch.token    = (whisper_token *  ) malloc(sizeof(whisper_token)    * (n_tokens));
    batch.pos      = (whisper_pos *)     malloc(sizeof(whisper_pos)      * (n_tokens));
    batch.n_seq_id = (int32_t *)         malloc(sizeof(int32_t)          * (n_tokens));
    batch.seq_id   = (whisper_seq_id **) malloc(sizeof(whisper_seq_id *) * (n_tokens + 1));
    for (int i = 0; i < n_tokens; ++i) {
        batch.seq_id[i] = (whisper_seq_id *) malloc(sizeof(whisper_seq_id)   * n_seq_max);
    }
    batch.seq_id[n_tokens] = nullptr;
    batch.logits   = (int8_t *)          malloc(sizeof(int8_t)           * n_tokens);

    return batch;
}

static void whisper_batch_free(struct whisper_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i]; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}

static void whisper_batch_prep_legacy(whisper_batch & batch, const whisper_token * tokens, int n_tokens, int n_past, int seq_id) {
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        if (tokens) {
            batch.token[i] = tokens[i];
        }
        batch.pos     [i]    = n_past + i;
        batch.n_seq_id[i]    = 1;
        batch.seq_id  [i][0] = seq_id;
        batch.logits  [i]    = 0;
    }
    batch.logits[n_tokens - 1] = 1;
}

// replace std::pair by using customized pair struct (reason: std::pair is very slow)
template<typename A, typename B>
struct whisper_pair {
    A first;
    B second;

    // Define a constructor that takes two arguments.
    whisper_pair(const A& a, const B& b) : first(a), second(b) {}
    // Define a constructor that takes no argument.
    whisper_pair() : first(A()), second(B()) {}
};

// ggml_backend_sched wrapper for whisper usage
struct whisper_sched {
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> meta;
};

static size_t whisper_sched_size(struct whisper_sched & allocr) {
    size_t size = allocr.meta.size();
    for (int i = 0; i < ggml_backend_sched_get_n_backends(allocr.sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(allocr.sched, i);
        size += ggml_backend_sched_get_buffer_size(allocr.sched, backend);
    }
    return size;
}

// measure the memory usage of a graph and prepare the allocr's internal data buffer
static bool whisper_sched_graph_init(struct whisper_sched & allocr, std::vector<ggml_backend_t> backends, std::function<struct ggml_cgraph *()> && get_graph) {
    auto & sched = allocr.sched;
    auto & meta  = allocr.meta;

    sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), WHISPER_MAX_NODES, false);

    meta.resize(ggml_tensor_overhead()*WHISPER_MAX_NODES + ggml_graph_overhead());

    // since there are dependencies between the different graphs,
    // we need to allocate them instead of only reserving to get the correct compute buffer size
    if (!ggml_backend_sched_alloc_graph(sched, get_graph())) {
        // failed to allocate the compute buffer
        WHISPER_LOG_ERROR("%s: failed to allocate the compute buffer\n", __func__);
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

// medium
// hparams: {
// 'n_mels': 80,
// 'n_vocab': 51864,
// 'n_audio_ctx': 1500,
// 'n_audio_state': 1024,
// 'n_audio_head': 16,
// 'n_audio_layer': 24,
// 'n_text_ctx': 448,
// 'n_text_state': 1024,
// 'n_text_head': 16,
// 'n_text_layer': 24
// }
//
// default hparams (Whisper tiny)
struct whisper_hparams {
    int32_t n_vocab       = 51864;
    int32_t n_audio_ctx   = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head  = 6;
    int32_t n_audio_layer = 4;
    int32_t n_text_ctx    = 448;
    int32_t n_text_state  = 384;
    int32_t n_text_head   = 6;
    int32_t n_text_layer  = 4;
    int32_t n_mels        = 80;
    int32_t ftype         = 1;
    float   eps           = 1e-5f;
};

// // audio encoding layer
// struct whisper_layer_encoder {
//     // encoder.blocks.*.attn_ln
//     struct ggml_tensor * attn_ln_0_w;  //self_attn_layer_norm_w
//     struct ggml_tensor * attn_ln_0_b;  //self_attn_layer_norm_b

//     // encoder.blocks.*.attn.out
//     struct ggml_tensor * attn_ln_1_w;  //final_layer_norm_w
//     struct ggml_tensor * attn_ln_1_b;  //final_layer_norm_b

//     // encoder.blocks.*.attn.query
//     struct ggml_tensor * attn_q_w;     //
//     struct ggml_tensor * attn_q_b;

//     // encoder.blocks.*.attn.key
//     struct ggml_tensor * attn_k_w;

//     // encoder.blocks.*.attn.value
//     struct ggml_tensor * attn_v_w;
//     struct ggml_tensor * attn_v_b;

//     // encoder.blocks.*.mlp_ln
//     struct ggml_tensor * mlp_ln_w;
//     struct ggml_tensor * mlp_ln_b;

//     // encoder.blocks.*.mlp.0
//     struct ggml_tensor * mlp_0_w;
//     struct ggml_tensor * mlp_0_b;

//     // encoder.blocks.*.mlp.2
//     struct ggml_tensor * mlp_1_w;
//     struct ggml_tensor * mlp_1_b;
// };

//lfr
// audio encoding layer
struct whisper_layer_encoder {
    // encoder.blocks.*.attn_ln
    struct ggml_tensor * self_attn_layer_norm_w;
    struct ggml_tensor * self_attn_layer_norm_b;

    // encoder.blocks.*.attn.out
    struct ggml_tensor * attn_o_w;
    struct ggml_tensor * attn_o_b;

    // encoder.blocks.*.attn.query
    struct ggml_tensor * attn_q_w;
    struct ggml_tensor * attn_q_b;

    // encoder.blocks.*.attn.key
    struct ggml_tensor * attn_k_w;

    // encoder.blocks.*.attn.value
    struct ggml_tensor * attn_v_w;
    struct ggml_tensor * attn_v_b;

    // encoder.blocks.*.mlp_ln
    struct ggml_tensor * final_layer_norm_w;
    struct ggml_tensor * final_layer_norm_b;

    // encoder.blocks.*.mlp.0
    struct ggml_tensor * fc1_w;
    struct ggml_tensor * fc1_b;

    // encoder.blocks.*.mlp.2
    struct ggml_tensor * fc2_w;
    struct ggml_tensor * fc2_b;
};


// token decoding layer
struct whisper_layer_decoder {
    // decoder.blocks.*.attn_ln
    struct ggml_tensor * attn_ln_0_w;
    struct ggml_tensor * attn_ln_0_b;

    // decoder.blocks.*.attn.out
    struct ggml_tensor * attn_ln_1_w;
    struct ggml_tensor * attn_ln_1_b;

    // decoder.blocks.*.attn.query
    struct ggml_tensor * attn_q_w;
    struct ggml_tensor * attn_q_b;

    // decoder.blocks.*.attn.key
    struct ggml_tensor * attn_k_w;

    // decoder.blocks.*.attn.value
    struct ggml_tensor * attn_v_w;
    struct ggml_tensor * attn_v_b;

    // decoder.blocks.*.cross_attn_ln
    struct ggml_tensor * cross_attn_ln_0_w;
    struct ggml_tensor * cross_attn_ln_0_b;

    // decoder.blocks.*.cross_attn.out
    struct ggml_tensor * cross_attn_ln_1_w;
    struct ggml_tensor * cross_attn_ln_1_b;

    // decoder.blocks.*.cross_attn.query
    struct ggml_tensor * cross_attn_q_w;
    struct ggml_tensor * cross_attn_q_b;

    // decoder.blocks.*.cross_attn.key
    struct ggml_tensor * cross_attn_k_w;

    // decoder.blocks.*.cross_attn.value
    struct ggml_tensor * cross_attn_v_w;
    struct ggml_tensor * cross_attn_v_b;

    // decoder.blocks.*.mlp_ln
    struct ggml_tensor * mlp_ln_w;
    struct ggml_tensor * mlp_ln_b;

    // decoder.blocks.*.mlp.0
    struct ggml_tensor * mlp_0_w;
    struct ggml_tensor * mlp_0_b;

    // decoder.blocks.*.mlp.2
    struct ggml_tensor * mlp_1_w;
    struct ggml_tensor * mlp_1_b;
};

struct whisper_kv_cell {
    whisper_pos pos = -1;

    std::set<whisper_seq_id> seq_id;

    bool has_seq_id(const whisper_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }
};

struct whisper_kv_cache {
    uint32_t head = 0;
    uint32_t size = 0;

    // computed before each graph build
    uint32_t n = 0;

    std::vector<whisper_kv_cell> cells;

    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx = nullptr;

    ggml_backend_buffer_t buffer = nullptr;
};

struct whisper_model {
    e_model type = MODEL_UNKNOWN;

    whisper_hparams hparams;
    whisper_filters filters;

    // encoder.positional_embedding
    struct ggml_tensor * e_pe;

    // encoder.conv1
    struct ggml_tensor * e_conv_1_w;
    struct ggml_tensor * e_conv_1_b;

    // encoder.conv2
    struct ggml_tensor * e_conv_2_w;
    struct ggml_tensor * e_conv_2_b;

    // encoder.ln_post
    struct ggml_tensor * e_ln_w;
    struct ggml_tensor * e_ln_b;

    // decoder.positional_embedding
    struct ggml_tensor * d_pe;

    // decoder.token_embedding
    struct ggml_tensor * d_te;

    // decoder.ln
    struct ggml_tensor * d_ln_w;
    struct ggml_tensor * d_ln_b;

    std::vector<whisper_layer_encoder> layers_encoder;
    std::vector<whisper_layer_decoder> layers_decoder;

    // ggml context that contains all the meta information about the model tensors
    struct ggml_context * ctx = nullptr;

    // the model backend data is read-only and can be shared between processors
    ggml_backend_buffer_t buffer = nullptr;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct whisper_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct whisper_grammar_candidate {
    whisper_token          id;
    const uint32_t       * code_points;
    whisper_partial_utf8   partial_utf8;
};

// [EXPERIMENTAL] Token-level timestamps with DTW
struct whisper_aheads_masks {
    std::vector<struct ggml_tensor *> m;    // One mask per text layer.
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

struct whisper_state {
    int64_t t_sample_us = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    int64_t t_batchd_us = 0;
    int64_t t_prompt_us = 0;
    int64_t t_mel_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_encode = 0; // number of encoder calls
    int32_t n_decode = 0; // number of decoder calls with n_tokens == 1  (text-generation)
    int32_t n_batchd = 0; // number of decoder calls with n_tokens <  16 (batch decoding)
    int32_t n_prompt = 0; // number of decoder calls with n_tokens >  1  (prompt encoding)
    int32_t n_fail_p = 0; // number of logprob threshold failures
    int32_t n_fail_h = 0; // number of entropy threshold failures

    // padded buffer for flash-attention
    whisper_kv_cache kv_pad;

    whisper_mel mel;

    whisper_batch batch;

    std::vector<ggml_backend_t> backends;

    // - stores meta info about the intermediate tensors into the `meta` buffers
    whisper_sched sched_conv;
    whisper_sched sched_encode;
    whisper_sched sched_cross;
    whisper_sched sched_decode;

    // result of the encoder
    struct ggml_tensor * embd_conv = nullptr;
    struct ggml_tensor * embd_enc  = nullptr;

    // helpers for GPU offloading
    std::vector<float> inp_mel;
    std::vector<float> inp_mask;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;

    int lang_id = 0; // english by default

    std::string path_model; // populated by whisper_init_from_file_with_params()

#ifdef WHISPER_USE_COREML
    whisper_coreml_context * ctx_coreml = nullptr;
#endif

#ifdef WHISPER_USE_OPENVINO
    whisper_openvino_context * ctx_openvino = nullptr;
#endif

    // [EXPERIMENTAL] token-level timestamps data
    int64_t t_beg  = 0;
    int64_t t_last = 0;

    whisper_token tid_last;

    std::vector<float> energy; // PCM signal energy

    // [EXPERIMENTAL] Token-level timestamps with DTW
    whisper_aheads_masks aheads_masks;
    ggml_tensor * aheads_cross_QKs = nullptr;
    std::vector<float> aheads_cross_QKs_data;

    // [EXPERIMENTAL] speed-up techniques
    int32_t exp_n_audio_ctx = 0; // 0 - use default
};

struct whisper_context {
    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    ggml_type wtype = ggml_type::GGML_TYPE_F16; // weight type (FP32 / FP16 / QX)
    ggml_type itype = ggml_type::GGML_TYPE_F16; // intermediate type (FP32 or FP16)

    whisper_context_params params;

    whisper_model model;
    whisper_vocab vocab;

    whisper_state * state = nullptr;

    std::string path_model; // populated by whisper_init_from_file_with_params()
};

struct whisper_global {
    // We save the log callback globally
    ggml_log_callback log_callback = whisper_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static whisper_global g_state;

template<typename T>
static void read_safe(whisper_model_loader * loader, T & dest) {
    loader->read(loader->context, &dest, sizeof(T));
    BYTESWAP_VALUE(dest);
}

static bool whisper_kv_cache_init(
             struct whisper_kv_cache & cache,
                      ggml_backend_t   backend,
                           ggml_type   wtype,
                             int64_t   n_text_state,
                             int64_t   n_text_layer,
                                 int   n_ctx) {
    const int64_t n_mem      = n_text_layer*n_ctx;
    const int64_t n_elements = n_text_state*n_mem;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    cache.head = 0;
    cache.size = n_ctx;

    cache.cells.clear();
    cache.cells.resize(n_ctx);

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        WHISPER_LOG_ERROR("%s: failed to allocate memory for the kv cache context\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);

    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (!cache.buffer) {
        WHISPER_LOG_ERROR("%s: failed to allocate memory for the kv cache\n", __func__);
        return false;
    }

    ggml_backend_buffer_clear(cache.buffer, 0);

    return true;
}

static void whisper_kv_cache_free(struct whisper_kv_cache & cache) {
    ggml_free(cache.ctx);
    ggml_backend_buffer_free(cache.buffer);
    cache.ctx = nullptr;
}

static bool whisper_kv_cache_find_slot(
           struct whisper_kv_cache & cache,
        const struct whisper_batch & batch) {
    const uint32_t n_ctx    = cache.size;
    const uint32_t n_tokens = batch.n_tokens;

    if (n_tokens > n_ctx) {
        WHISPER_LOG_ERROR("%s: n_tokens=%d > n_ctx=%d\n", __func__, n_tokens, n_ctx);
        return false;
    }

    uint32_t n_tested = 0;

    while (true) {
        if (cache.head + n_tokens > n_ctx) {
            n_tested += n_ctx - cache.head;
            cache.head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                found = false;
                cache.head += i + 1;
                n_tested   += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= n_ctx) {
            //WHISPER_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return false;
        }
    }

    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];

        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
        }
    }

    return true;
}

// find how many cells are currently in use
static int32_t whisper_kv_cache_cell_max(const struct whisper_kv_cache & cache) {
    for (uint32_t i = cache.size - 1; i > 0; --i) {
        if (cache.cells[i].pos >= 0 && !cache.cells[i].seq_id.empty()) {
            return i + 1;
        }
    }

    return 1;
}

static void whisper_kv_cache_clear(struct whisper_kv_cache & cache) {
    for (int32_t i = 0; i < (int32_t) cache.size; ++i) {
        cache.cells[i].pos = -1;
        cache.cells[i].seq_id.clear();
    }
    cache.head = 0;

    ggml_backend_buffer_clear(cache.buffer, 0);
}

static void whisper_kv_cache_seq_rm(
        struct whisper_kv_cache & cache,
                 whisper_seq_id   seq_id,
                    whisper_pos   p0,
                    whisper_pos   p1) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<whisper_pos>::max();

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            if (seq_id < 0) {
                cache.cells[i].seq_id.clear();
            } else if (cache.cells[i].has_seq_id(seq_id)) {
                cache.cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cache.cells[i].seq_id.empty()) {
                cache.cells[i].pos = -1;
                if (new_head == cache.size) new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size) cache.head = new_head;
}

static void whisper_kv_cache_seq_cp(
        struct whisper_kv_cache & cache,
                 whisper_seq_id   seq_id_src,
                 whisper_seq_id   seq_id_dst,
                    whisper_pos   p0,
                    whisper_pos   p1) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<whisper_pos>::max();

    cache.head = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id_src) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

static uint32_t whisper_kv_cache_get_padding(const struct whisper_context & wctx) {
    if (!wctx.params.flash_attn || !wctx.params.use_gpu) {
        return 1u;
    }

#ifdef GGML_USE_METAL
    if (wctx.params.use_gpu) {
        return 32u;
    }
#endif

#ifdef GGML_USE_CUDA
    if (wctx.params.use_gpu) {
        return 256u;
    }
#endif

    return 1u;
}

// [EXPERIMENTAL] Token-level timestamps with DTW
static bool aheads_masks_init(
        const whisper_context_params & cparams,
               const whisper_hparams & hparams,
         struct whisper_aheads_masks & aheads_masks,
                      ggml_backend_t   backend) {

    const int32_t n_text_layer = hparams.n_text_layer;
    const int32_t n_head = hparams.n_text_head;

    // Sanity checks
    if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE) {
        WHISPER_LOG_ERROR("%s: dtw_aheads_preset should be != DTW_AHEADS_NONE\n", __func__);
        return false;
    } else if (cparams.dtw_aheads_preset == WHISPER_AHEADS_N_TOP_MOST) {
        if (cparams.dtw_n_top > n_text_layer || cparams.dtw_n_top <= 0) {
            WHISPER_LOG_ERROR("%s: dtw_n_top must be between %d and %d for this model.", __func__, 1, n_text_layer);
            return false;
        }
    } else {
        const auto aheads = cparams.dtw_aheads_preset == WHISPER_AHEADS_CUSTOM ? cparams.dtw_aheads : g_aheads.at(cparams.dtw_aheads_preset);
        if (cparams.dtw_aheads_preset == WHISPER_AHEADS_CUSTOM) {
            if (aheads.n_heads == 0) {
                WHISPER_LOG_ERROR("%s: dtw_aheads.n_heads should be > 0", __func__);
                return false;
            }
            if (aheads.heads == NULL) {
                WHISPER_LOG_ERROR("%s: dtw_aheads.heads unset", __func__);
                return false;
            }
        }
        for (size_t i = 0; i < aheads.n_heads; ++i) {
            if (aheads.heads[i].n_text_layer >= n_text_layer) {
                WHISPER_LOG_ERROR("%s: tried to set alignment head on text layer %d, but model only has %d text layers", __func__, aheads.heads[i].n_text_layer + 1, n_text_layer);
                return false;
            }
            if (aheads.heads[i].n_text_layer < 0) {
                WHISPER_LOG_ERROR("%s: tried to set alignment head on text layer < 0", __func__);
                return false;
            }
            if (aheads.heads[i].n_head >= n_head) {
                WHISPER_LOG_ERROR("%s: tried to set alignment head on head %d, but model only has %d heads", __func__, aheads.heads[i].n_head + 1, n_head);
                return false;
            }
            if (aheads.heads[i].n_head < 0) {
                WHISPER_LOG_ERROR("%s: tried to set alignment head on head < 0", __func__);
                return false;
            }
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ (size_t) static_cast<size_t>(n_text_layer)*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    aheads_masks.ctx = ggml_init(params);

    if (!aheads_masks.ctx) {
        WHISPER_LOG_ERROR("%s: failed to allocate memory for the aheads_masks context\n", __func__);
        return false;
    }

    for (int64_t il = 0; il < n_text_layer; ++il) {
        auto aheads = get_alignment_heads_by_layer(cparams, il, n_text_layer, n_head);
        if (!aheads.empty()) {
            aheads_masks.m.push_back(ggml_new_tensor_2d(aheads_masks.ctx, GGML_TYPE_F32, n_head, aheads.size()));
        } else {
            aheads_masks.m.push_back(nullptr);
        }
    }

    aheads_masks.buffer = ggml_backend_alloc_ctx_tensors(aheads_masks.ctx, backend);
    if (!aheads_masks.buffer) {
        WHISPER_LOG_ERROR("%s: failed to allocate memory for aheads_masks\n", __func__);
        return false;
    }

    // Set data on mask tensors
    // Since this must be backend agnostic, we write our desired values on mask_data,
    // and send it to backend with ggml_backend_tensor_set.
    // Each mask in N_HEADS*N_ALIGNMENT_HEADS, one per text layer containing alignment
    // heads. Each row of the mask "marks" one alignment head. E.g. if some text layer
    // has a total of 10 heads and of those, heads 0,5,6 are alignment heads, the mask
    // should read:
    // 1 0 0 0 0 0 0 0 0 0
    // 0 0 0 0 0 1 0 0 0 0
    // 0 0 0 0 0 0 1 0 0 0
    std::vector<float> mask_data;
    for (int64_t il = 0; il < n_text_layer; ++il) {
        if (aheads_masks.m[il] != nullptr) {
            auto aheads = get_alignment_heads_by_layer(cparams, il, n_text_layer, n_head);

            size_t data_size = aheads_masks.m[il]->ne[0] * aheads_masks.m[il]->ne[1];
            size_t data_size_bytes = data_size * sizeof(float);
            mask_data.resize(data_size);

            std::fill(mask_data.begin(), mask_data.end(), 0);
            for (size_t ih = 0; ih < aheads.size(); ++ih) {
                size_t pos = (aheads[ih] + (ih * aheads_masks.m[il]->ne[0]));
                mask_data[pos] = 1.0f;
            }

            ggml_backend_tensor_set(aheads_masks.m[il], mask_data.data(), 0, data_size_bytes);
        }
    }

    if (aheads_masks.m.empty()) {
        WHISPER_LOG_ERROR("%s: \n", __func__);
        return false;
    }

    return true;
}

static void aheads_masks_free(struct whisper_aheads_masks & aheads_masks) {
    ggml_free(aheads_masks.ctx);
    ggml_backend_buffer_free(aheads_masks.buffer);
    aheads_masks.ctx = nullptr;
}

static size_t aheads_masks_nbytes(struct whisper_aheads_masks & aheads_masks) {
    size_t size = 0;
    for (size_t i = 0; i < aheads_masks.m.size(); ++i) {
        if (aheads_masks.m[i] != nullptr)
            size += ggml_nbytes(aheads_masks.m[i]);
    }
    return size;
}

static ggml_backend_t whisper_backend_init_gpu(const whisper_context_params & params) {
    ggml_backend_t result = NULL;

    ggml_log_set(g_state.log_callback, g_state.log_callback_user_data);

#ifdef GGML_USE_CUDA
    if (params.use_gpu) {
        WHISPER_LOG_INFO("%s: using CUDA backend\n", __func__);
        result = ggml_backend_cuda_init(params.gpu_device);
        if (!result) {
            WHISPER_LOG_ERROR("%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (params.use_gpu) {
        WHISPER_LOG_INFO("%s: using Metal backend\n", __func__);
        result = ggml_backend_metal_init();
        if (!result) {
            WHISPER_LOG_ERROR("%s: ggml_backend_metal_init() failed\n", __func__);
        } else if (!ggml_backend_metal_supports_family(result, 7)) {
            WHISPER_LOG_ERROR("%s: Metal GPU does not support family 7 - falling back to CPU\n", __func__);
            ggml_backend_free(result);
            result = NULL;
        }
    }
#endif

#ifdef GGML_USE_SYCL
    if (params.use_gpu) {
        WHISPER_LOG_INFO("%s: using SYCL backend\n", __func__);
        result = ggml_backend_sycl_init(params.gpu_device);
        if (!result) {
            WHISPER_LOG_ERROR("%s: ggml_backend_sycl_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_VULKAN
    if (params.use_gpu) {
        WHISPER_LOG_INFO("%s: using Vulkan backend\n", __func__);
        result = ggml_backend_vk_init(params.gpu_device);
        if (!result) {
            WHISPER_LOG_ERROR("%s: ggml_backend_vk_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_CANN
    if (params.use_gpu) {
        WHISPER_LOG_INFO("%s: using CANN backend\n", __func__);
        result = ggml_backend_cann_init(params.gpu_device);
        if (!result) {
            WHISPER_LOG_ERROR("%s: ggml_backend_cann_init() failed\n", __func__);
        }
    }
#endif

    GGML_UNUSED(params);

    return result;
}

static std::vector<ggml_backend_t> whisper_backend_init(const whisper_context_params & params) {
    std::vector<ggml_backend_t> result;

    ggml_backend_t backend_gpu = whisper_backend_init_gpu(params);

    if (backend_gpu) {
        result.push_back(backend_gpu);
    }

#ifdef GGML_USE_BLAS
    {
        WHISPER_LOG_INFO("%s: using BLAS backend\n", __func__);
        ggml_backend_t backend_blas = ggml_backend_blas_init();
        if (!backend_blas) {
            WHISPER_LOG_ERROR("%s: ggml_backend_blas_init() failed\n", __func__);
        } else {
            result.push_back(backend_blas);
        }
    }
#endif

    GGML_UNUSED(params);

    result.push_back(ggml_backend_cpu_init());

    return result;
}

static ggml_backend_buffer_type_t whisper_default_buffer_type(const whisper_context_params & params) {
    ggml_backend_buffer_type_t result = nullptr;

    params.use_gpu || (result = ggml_backend_cpu_buffer_type());

#ifdef GGML_USE_CUDA
    result || (result = ggml_backend_cuda_buffer_type(params.gpu_device));
#endif

#ifdef GGML_USE_METAL
    result || (result = ggml_backend_metal_buffer_type());
#endif

#ifdef GGML_USE_SYCL
    result || (result = ggml_backend_sycl_buffer_type(params.gpu_device));
#endif

#ifdef GGML_USE_VULKAN
    result || (result = ggml_backend_vk_buffer_type(params.gpu_device));
#endif

#ifdef GGML_USE_CANN
    result || (result == ggml_backend_cann_buffer_type(params.gpu_device));
#endif

    result || (result = ggml_backend_cpu_buffer_type());

    return result;
}

// load the model from a ggml file
//
// file format:
//
//   - hparams
//   - pre-computed mel filters
//   - vocab
//   - weights
//
// see the convert-pt-to-ggml.py script for details
//
static bool whisper_model_load(struct whisper_model_loader * loader, whisper_context & wctx) {
    WHISPER_LOG_INFO("%s: loading model\n", __func__);

    const int64_t t_start_us = ggml_time_us();

    wctx.t_start_us = t_start_us;

    auto & model = wctx.model;
    auto & vocab = wctx.vocab;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC) {
            WHISPER_LOG_ERROR("%s: invalid model data (bad magic)\n", __func__);
            return false;
        }
    }

    //load hparams
    {
        auto & hparams = model.hparams;

        read_safe(loader, hparams.n_vocab);
        read_safe(loader, hparams.n_audio_ctx);
        read_safe(loader, hparams.n_audio_state);
        read_safe(loader, hparams.n_audio_head);
        read_safe(loader, hparams.n_audio_layer);
        read_safe(loader, hparams.n_text_ctx);
        read_safe(loader, hparams.n_text_state);
        read_safe(loader, hparams.n_text_head);
        read_safe(loader, hparams.n_text_layer);
        read_safe(loader, hparams.n_mels);
        read_safe(loader, hparams.ftype);

        assert(hparams.n_text_state == hparams.n_audio_state);

        std::string mver = "";

        if (hparams.n_audio_layer == 4) {
            model.type = e_model::MODEL_TINY;
        }

        if (hparams.n_audio_layer == 6) {
            model.type = e_model::MODEL_BASE;
        }

        if (hparams.n_audio_layer == 12) {
            model.type = e_model::MODEL_SMALL;
        }

        if (hparams.n_audio_layer == 24) {
            model.type = e_model::MODEL_MEDIUM;
        }

        if (hparams.n_audio_layer == 32) {
            model.type = e_model::MODEL_LARGE;

            if (hparams.n_vocab == 51866) {
                mver = " v3";
            }
        }

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;

        // for the big tensors, we have the option to store the data in 16-bit floats or quantized
        // in order to save memory and also to speed up the computation
        wctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
        if (wctx.wtype == GGML_TYPE_COUNT) {
            WHISPER_LOG_ERROR("%s: invalid model (bad ftype value %d)\n", __func__, model.hparams.ftype);
            return false;
        }

        WHISPER_LOG_INFO("%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        WHISPER_LOG_INFO("%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        WHISPER_LOG_INFO("%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        WHISPER_LOG_INFO("%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        WHISPER_LOG_INFO("%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        WHISPER_LOG_INFO("%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
        WHISPER_LOG_INFO("%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
        WHISPER_LOG_INFO("%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
        WHISPER_LOG_INFO("%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
        WHISPER_LOG_INFO("%s: n_mels        = %d\n", __func__, hparams.n_mels);
        WHISPER_LOG_INFO("%s: ftype         = %d\n", __func__, model.hparams.ftype);
        WHISPER_LOG_INFO("%s: qntvr         = %d\n", __func__, qntvr);
        WHISPER_LOG_INFO("%s: type          = %d (%s%s)\n", __func__, model.type, g_model_name.at(model.type).c_str(), mver.c_str());
    }

    // load mel filters
    {
        auto & filters = wctx.model.filters;

        read_safe(loader, filters.n_mel);
        read_safe(loader, filters.n_fft);

        filters.data.resize(filters.n_mel * filters.n_fft);
        loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));
        BYTESWAP_FILTERS(filters);
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        read_safe(loader, n_vocab);

        //if (n_vocab != model.hparams.n_vocab) {
        //    WHISPER_LOG_ERROR("%s: invalid model file '%s' (bad vocab size %d != %d)\n",
        //            __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
        //    return false;
        //}

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            read_safe(loader, len);

            if (len > 0) {
                tmp.resize(len);
                loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
                word.assign(&tmp[0], tmp.size());
            } else {
                // seems like we have an empty-string token in multi-language models (i = 50256)
                //WHISPER_LOG_WARN("%s: warning: empty-string token in vocab, i = %d\n", __func__, i);
                word = "";
            }

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;

            //printf("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
        }

        vocab.n_vocab = model.hparams.n_vocab;
        if (vocab.is_multilingual()) {
            vocab.token_eot++;
            vocab.token_sot++;

            // account for variable number of language tokens
            const int dt = vocab.num_languages() - 98;

            vocab.token_translate  += dt;
            vocab.token_transcribe += dt;
            vocab.token_solm       += dt;
            vocab.token_prev       += dt;
            vocab.token_nosp       += dt;
            vocab.token_not        += dt;
            vocab.token_beg        += dt;
        }

        if (n_vocab < model.hparams.n_vocab) {
            WHISPER_LOG_INFO("%s: adding %d extra tokens\n", __func__, model.hparams.n_vocab - n_vocab);
            for (int i = n_vocab; i < model.hparams.n_vocab; i++) {
                if (i > vocab.token_beg) {
                    word = "[_TT_" + std::to_string(i - vocab.token_beg) + "]";
                } else if (i == vocab.token_eot) {
                    word = "[_EOT_]";
                } else if (i == vocab.token_sot) {
                    word = "[_SOT_]";
                } else if (i == vocab.token_translate) {
                    word = "[_TRANSLATE_]";
                } else if (i == vocab.token_transcribe) {
                    word = "[_TRANSCRIBE_]";
                } else if (i == vocab.token_solm) {
                    word = "[_SOLM_]";
                } else if (i == vocab.token_prev) {
                    word = "[_PREV_]";
                } else if (i == vocab.token_nosp) {
                    word = "[_NOSP_]";
                } else if (i == vocab.token_not) {
                    word = "[_NOT_]";
                } else if (i == vocab.token_beg) {
                    word = "[_BEG_]";
                } else if (i > vocab.token_sot && i <= vocab.token_sot + vocab.num_languages()) {
                    word = "[_LANG_" + std::string(whisper_lang_str(i - vocab.token_sot - 1)) + "]";
                } else {
                    word = "[_extra_token_" + std::to_string(i) + "]";
                }
                vocab.token_to_id[word] = i;
                vocab.id_to_token[i] = word;
            }
        }

        WHISPER_LOG_INFO("%s: n_langs       = %d\n", __func__, vocab.num_languages());
    }

    const ggml_type wtype = wctx.wtype;
    const ggml_type vtype = wctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16; // conv type

    // create the ggml context
    {
        const auto & hparams = model.hparams;

        const int n_audio_layer = hparams.n_audio_layer;
        const int n_text_layer  = hparams.n_text_layer;
        //lfr
        //const size_t n_tensors = 10 /* input */ + 15 + 15*n_audio_layer + 24*n_text_layer;
        const size_t n_tensors = 7 + 15*n_audio_layer + 24*n_text_layer;

        struct ggml_init_params params = {
            /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            WHISPER_LOG_ERROR("%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare tensors for the weights
    {
        auto & ctx = model.ctx;

        const auto & hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;

        const int n_audio_ctx   = hparams.n_audio_ctx;
        const int n_audio_state = hparams.n_audio_state;
        const int n_audio_layer = hparams.n_audio_layer;

        const int n_text_ctx   = hparams.n_text_ctx;
        const int n_text_state = hparams.n_text_state;
        const int n_text_layer = hparams.n_text_layer;

        const int n_mels = hparams.n_mels;

        model.layers_encoder.resize(n_audio_layer);
        model.layers_decoder.resize(n_text_layer);

        // encoder
        {
            model.e_pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_state, n_audio_ctx);

            model.e_conv_1_w     = ggml_new_tensor_3d(ctx, vtype,         3, n_mels,     n_audio_state);
            model.e_conv_1_b     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,         1,     n_audio_state);

            model.e_conv_2_w     = ggml_new_tensor_3d(ctx, vtype,         3, n_audio_state, n_audio_state);
            model.e_conv_2_b     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,                1, n_audio_state);

            model.e_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
            model.e_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

            // map by name
            model.tensors["embed_positions.weight"] = model.e_pe;

            model.tensors["conv1.weight"]         = model.e_conv_1_w;
            model.tensors["conv1.bias"]           = model.e_conv_1_b;

            model.tensors["conv2.weight"]         = model.e_conv_2_w;
            model.tensors["conv2.bias"]           = model.e_conv_2_b;

            model.tensors["layer_norm.weight"]       = model.e_ln_w;
            model.tensors["layer_norm.bias"]         = model.e_ln_b;

            for (int i = 0; i < n_audio_layer; ++i) {
                auto & layer = model.layers_encoder[i];

                layer.final_layer_norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);
                layer.final_layer_norm_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                layer.fc1_w     = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, 4*n_audio_state);
                layer.fc1_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_audio_state);

                layer.fc2_w     = ggml_new_tensor_2d(ctx, wtype,         4*n_audio_state, n_audio_state);
                layer.fc2_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                layer.self_attn_layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);
                layer.self_attn_layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                layer.attn_q_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
                layer.attn_q_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                layer.attn_k_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);

                layer.attn_v_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
                layer.attn_v_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                layer.attn_o_w = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
                layer.attn_o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                // map by name
                model.tensors["layers." + std::to_string(i) + ".final_layer_norm.weight"]     = layer.final_layer_norm_w;
                model.tensors["layers." + std::to_string(i) + ".final_layer_norm.bias"]       = layer.final_layer_norm_b;

                model.tensors["layers." + std::to_string(i) + ".fc1.weight"]      = layer.fc1_w;
                model.tensors["layers." + std::to_string(i) + ".fc1.bias"]        = layer.fc1_b;

                model.tensors["layers." + std::to_string(i) + ".fc2.weight"]      = layer.fc2_w;
                model.tensors["layers." + std::to_string(i) + ".fc2.bias"]        = layer.fc2_b;

                model.tensors["layers." + std::to_string(i) + ".self_attn_layer_norm.weight"]    = layer.self_attn_layer_norm_w;
                model.tensors["layers." + std::to_string(i) + ".self_attn_layer_norm.bias"]      = layer.self_attn_layer_norm_b;

                model.tensors["layers." + std::to_string(i) + ".self_attn.q_proj.weight"] = layer.attn_q_w;
                model.tensors["layers." + std::to_string(i) + ".self_attn.q_proj.bias"]   = layer.attn_q_b;

                model.tensors["layers." + std::to_string(i) + ".self_attn.k_proj.weight"]   = layer.attn_k_w;

                model.tensors["layers." + std::to_string(i) + ".self_attn.v_proj.weight"] = layer.attn_v_w;
                model.tensors["layers." + std::to_string(i) + ".self_attn.v_proj.bias"]   = layer.attn_v_b;

                model.tensors["layers." + std::to_string(i) + ".self_attn.out_proj.weight"]   = layer.attn_o_w;
                model.tensors["layers." + std::to_string(i) + ".self_attn.out_proj.bias"]     = layer.attn_o_b;
            }
        }

        // decoder
        // {
        //     model.d_pe   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_text_state, n_text_ctx);

        //     model.d_te   = ggml_new_tensor_2d(ctx, wtype,         n_text_state, n_vocab);

        //     model.d_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
        //     model.d_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

        //     // map by name
        //     model.tensors["decoder.positional_embedding"]   = model.d_pe;

        //     model.tensors["decoder.token_embedding.weight"] = model.d_te;

        //     model.tensors["decoder.ln.weight"]              = model.d_ln_w;
        //     model.tensors["decoder.ln.bias"]                = model.d_ln_b;

        //     for (int i = 0; i < n_text_layer; ++i) {
        //         auto & layer = model.layers_decoder[i];

        //         layer.mlp_ln_w          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);
        //         layer.mlp_ln_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

        //         layer.mlp_0_w           = ggml_new_tensor_2d(ctx, wtype,           n_text_state, 4*n_text_state);
        //         layer.mlp_0_b           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_text_state);

        //         layer.mlp_1_w           = ggml_new_tensor_2d(ctx, wtype,         4*n_text_state, n_text_state);
        //         layer.mlp_1_b           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

        //         layer.attn_ln_0_w       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);
        //         layer.attn_ln_0_b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

        //         layer.attn_q_w          = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
        //         layer.attn_q_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

        //         layer.attn_k_w          = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);

        //         layer.attn_v_w          = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
        //         layer.attn_v_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

        //         layer.attn_ln_1_w       = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
        //         layer.attn_ln_1_b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

        //         layer.cross_attn_ln_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);
        //         layer.cross_attn_ln_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

        //         layer.cross_attn_q_w    = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
        //         layer.cross_attn_q_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

        //         layer.cross_attn_k_w    = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);

        //         layer.cross_attn_v_w    = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
        //         layer.cross_attn_v_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

        //         layer.cross_attn_ln_1_w = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
        //         layer.cross_attn_ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

        //         // map by name
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".mlp_ln.weight"]           = layer.mlp_ln_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".mlp_ln.bias"]             = layer.mlp_ln_b;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.0.weight"]            = layer.mlp_0_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.0.bias"]              = layer.mlp_0_b;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.2.weight"]            = layer.mlp_1_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.2.bias"]              = layer.mlp_1_b;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".attn_ln.weight"]          = layer.attn_ln_0_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".attn_ln.bias"]            = layer.attn_ln_0_b;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".attn.query.weight"]       = layer.attn_q_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".attn.query.bias"]         = layer.attn_q_b;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".attn.key.weight"]         = layer.attn_k_w;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".attn.value.weight"]       = layer.attn_v_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".attn.value.bias"]         = layer.attn_v_b;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".attn.out.weight"]         = layer.attn_ln_1_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".attn.out.bias"]           = layer.attn_ln_1_b;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn_ln.weight"]    = layer.cross_attn_ln_0_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn_ln.bias"]      = layer.cross_attn_ln_0_b;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.query.weight"] = layer.cross_attn_q_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.query.bias"]   = layer.cross_attn_q_b;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.key.weight"]   = layer.cross_attn_k_w;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.value.weight"] = layer.cross_attn_v_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.value.bias"]   = layer.cross_attn_v_b;

        //         model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.out.weight"]   = layer.cross_attn_ln_1_w;
        //         model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.out.bias"]     = layer.cross_attn_ln_1_b;
        //     }
        // }
    }

    // allocate tensors in the backend buffers
    model.buffer = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx, whisper_default_buffer_type(wctx.params));
    if (!model.buffer) {
        WHISPER_LOG_ERROR("%s: failed to allocate memory for the model\n", __func__);
        return false;
    }

    size_t size_main = ggml_backend_buffer_get_size(model.buffer);
    WHISPER_LOG_INFO("%s: %8s total size = %8.2f MB\n", __func__, ggml_backend_buffer_name(model.buffer), size_main / 1e6);

    // load weights
    {
        size_t total_size = 0;

        model.n_loaded = 0;

        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            read_safe(loader, n_dims);
            read_safe(loader, length);
            read_safe(loader, ttype);

            if (loader->eof(loader->context)) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[4] = { 1, 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                read_safe(loader, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> tmp(length); // create a buffer
            loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
            name.assign(&tmp[0], tmp.size());

            if (model.tensors.find(name) == model.tensors.end()) {
                WHISPER_LOG_ERROR("%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];

            if (ggml_nelements(tensor) != nelements) {
                WHISPER_LOG_ERROR("%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                WHISPER_LOG_ERROR("%s: shape: [%d, %d, %d], expected: [%d, %d, %d]\n",
                        __func__, ne[0], ne[1], ne[2], (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2]);
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
                WHISPER_LOG_ERROR("%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d], expected [%d, %d, %d]\n",
                        __func__, name.data(), (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2], ne[0], ne[1], ne[2]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                WHISPER_LOG_ERROR("%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            //ggml_backend_t backend = wctx.backend;

            //printf("%s: [%5.5s] %s\n", __func__, ggml_backend_name(backend), name.c_str());

            if (ggml_backend_buffer_is_host(model.buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                loader->read(loader->context, tensor->data, ggml_nbytes(tensor));
                BYTESWAP_TENSOR(tensor);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));

                loader->read(loader->context, read_buf.data(), read_buf.size());

                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            //printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype), ggml_nbytes(tensor)/1e6);
            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        WHISPER_LOG_INFO("%s: model size    = %7.2f MB\n", __func__, total_size/1e6);

        if (model.n_loaded == 0) {
            WHISPER_LOG_WARN("%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
        } else if (model.n_loaded != (int) model.tensors.size()) {
            WHISPER_LOG_ERROR("%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
            return false;
        }
    }

    ggml_backend_buffer_set_usage(model.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    wctx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}

static bool whisper_encode_external(const whisper_state & wstate) {
    GGML_UNUSED(wstate);

#ifndef WHISPER_USE_COREML
    const bool use_coreml = false;
#else
    const bool use_coreml = wstate.ctx_coreml != nullptr;
#endif

#ifndef WHISPER_USE_OPENVINO
    const bool use_openvino = false;
#else
    const bool use_openvino = wstate.ctx_openvino != nullptr;
#endif

    return use_coreml || use_openvino;
}

static struct ggml_cgraph * whisper_build_graph_conv(
        whisper_context & wctx,
          whisper_state & wstate) {
    const auto & model   = wctx.model;
    const auto & hparams = model.hparams;

    const int n_ctx   = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state; GGML_UNUSED(n_state);

    const int n_mels = hparams.n_mels;

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_conv.meta.size(),
        /*.mem_buffer =*/ wstate.sched_conv.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2*n_ctx, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    struct ggml_tensor * cur = nullptr;

    if (!whisper_encode_external(wstate)) {
        // convolution + gelu
        {
            cur = ggml_conv_1d_ph(ctx0, model.e_conv_1_w, mel, 1, 1);
            cur = ggml_add(ctx0, cur, model.e_conv_1_b);

            cur = ggml_gelu(ctx0, cur);

            cur = ggml_conv_1d_ph(ctx0, model.e_conv_2_w, cur, 2, 1);
            cur = ggml_add(ctx0, cur, model.e_conv_2_b);

            cur = ggml_gelu(ctx0, cur);
        }

        ggml_set_name(cur, "embd_conv");
        wstate.embd_conv = cur;
    } else {
        ggml_build_forward_expand(gf, mel);

        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx);
        ggml_set_input(cur); // the external encoder will write into this tensor

        ggml_set_name(cur, "embd_enc");
        wstate.embd_enc = cur;
    }

    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * whisper_build_graph_encoder(
        whisper_context & wctx,
          whisper_state & wstate) {
    const auto & model   = wctx.model;
    const auto & hparams = model.hparams;

    const int n_ctx   = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    const int n_head  = hparams.n_audio_head;
    const int n_layer = hparams.n_audio_layer;

    const int n_state_head = n_state/n_head;

    // auto & kv_pad = wstate.kv_pad;

    // WHISPER_ASSERT(!!kv_pad.ctx);

    // const int n_ctx_pad = GGML_PAD(n_ctx, 256);

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_encode.meta.size(),
        /*.mem_buffer =*/ wstate.sched_encode.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, WHISPER_MAX_NODES, false);

    struct ggml_tensor * cur = ggml_view_tensor(ctx0, wstate.embd_conv);

    const float KQscale = 1.0f/sqrtf(float(n_state_head));

    // ===================================================================
    // NOTE: experimenting with partial evaluation of the encoder (ignore)
    //static int iter = -1;
    //const int n_iter = 1500/n_ctx;

    //iter = (iter + 1) % n_iter;

    //if (iter == 0) {
    //    memset(model.memory_cross_k->data, 0, ggml_nbytes(model.memory_cross_k));
    //    memset(model.memory_cross_v->data, 0, ggml_nbytes(model.memory_cross_v));
    //}

    static int iter = 0;

    const size_t e_pe_stride = model.e_pe->ne[0]*ggml_element_size(model.e_pe);
    const size_t e_pe_offset = model.e_pe->ne[0]*ggml_element_size(model.e_pe)*n_ctx*iter;

    struct ggml_tensor * e_pe = ggml_view_2d(ctx0, model.e_pe, model.e_pe->ne[0], n_ctx, e_pe_stride, e_pe_offset);
    cur = ggml_add(ctx0, e_pe, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));

    // ===================================================================

    // original:
    //cur = ggml_add(ctx0, model.e_pe, ggml_transpose(ctx0, cur));

    struct ggml_tensor * inpL = cur;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers_encoder[il];

        // norm
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);

            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0, cur, layer.self_attn_layer_norm_w),
                    layer.self_attn_layer_norm_b);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0,
                    layer.attn_q_w,
                    cur);

            Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);

            //Qcur = ggml_scale(ctx0, Qcur, pow(float(n_state_head), -0.25));

            // note: no bias for Key
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0,
                    layer.attn_k_w,
                    cur);

            //Kcur = ggml_scale(ctx0, Kcur, pow(float(n_state_head), -0.25));

            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0,
                    layer.attn_v_w,
                    cur);

            Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);

            // ------

            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_scale(ctx0, ggml_reshape_3d(ctx0, Qcur, n_state_head, n_head, n_ctx), KQscale), 
                        0, 2, 1, 3);

            if (wctx.params.flash_attn) {
                // ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, ggml_view_1d(ctx0, kv_pad.k, n_ctx*n_state, 0)));
                // ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, ggml_view_1d(ctx0, kv_pad.v, n_ctx*n_state, 0)));

                // struct ggml_tensor * K =
                //     ggml_view_3d(ctx0, kv_pad.k,
                //             n_state_head, n_ctx_pad, n_head,
                //             ggml_element_size(kv_pad.k)*n_state,
                //             ggml_element_size(kv_pad.k)*n_state_head,
                //             0);

                // struct ggml_tensor * V =
                //     ggml_view_3d(ctx0, kv_pad.v,
                //             n_state_head, n_ctx_pad, n_head,
                //             ggml_element_size(kv_pad.v)*n_state,
                //             ggml_element_size(kv_pad.v)*n_state_head,
                //             0);

                // cur = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, KQscale, 0.0f, 0.0f);

                // cur = ggml_reshape_2d(ctx0, cur, n_state, n_ctx);
                ;
            } else {
                struct ggml_tensor * K =
                    ggml_permute(ctx0,
                            //ggml_cast(ctx0,
                                ggml_reshape_3d(ctx0, Kcur, n_state_head, n_head, n_ctx),
                            //    wctx.itype),
                            0, 2, 1, 3);

                // K * Q
                struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

                struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ);//ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);

                struct ggml_tensor * V =
                    //ggml_cast(ctx0,
                        ggml_cont(ctx0, 
                            ggml_permute(ctx0,
                                ggml_reshape_3d(ctx0,
                                    Vcur,
                                    n_state_head, n_head, n_ctx),
                                1, 2, 0, 3));
                            //wctx.itype);

                struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

                struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

                cur = ggml_cont_2d(ctx0, KQV_merged, n_state, n_ctx);
            }
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0,
                    layer.attn_o_w,
                    cur);

            cur = ggml_add(ctx0, cur, layer.attn_o_b);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_add(ctx0,
                        ggml_mul(ctx0, cur, layer.final_layer_norm_w),
                        layer.final_layer_norm_b);
            }

            // fully connected
            cur = ggml_mul_mat(ctx0,
                    layer.fc1_w,
                    cur);

            cur = ggml_add(ctx0, cur, layer.fc1_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0,
                    layer.fc2_w,
                    cur);

            cur = ggml_add(ctx0, cur, layer.fc2_b);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;
    //lfr
    //permute
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);

    //avg pooler lfr
    {
        cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 2, 2, 0);
    }

    //lfr
    //permute
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);

    // norm
    {
        cur = ggml_norm(ctx0, cur, hparams.eps);

        // cur = ln_f_g*cur + ln_f_b
        cur = ggml_add(ctx0,
                ggml_mul(ctx0, cur, model.e_ln_w),
                model.e_ln_b);
    }

    ggml_build_forward_expand(gf, cur);

    wstate.embd_enc = cur;

    //ggml_graph_print(gf);
    ///ggml_graph_dump_dot(gf, NULL, "./debug.dot");


    ////////////////////////////////////////////////////////////////////////////

    //printf("%s: used_mem = %f MB, %f MB, %f MB %f MB %f MB\n", __func__,
    //        ggml_used_mem(ctx0)/1e6,
    //        wstate.get_buf_max_mem(0)/1e6,
    //        wstate.get_buf_max_mem(1)/1e6,
    //        wstate.get_buf_max_mem(2)/1e6,
    //        wstate.get_buf_max_mem(3)/1e6);

    ggml_free(ctx0);

    return gf;
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {

    // fprintf(stderr, "%s\n",t->name);

    // if (t->op == GGML_OP_CONT) {
    //     int a = 9;
    // }
    if (ask) {
        return true; // Always retrieve data
    }



    return true;
}

// evaluate the encoder with the given state
//
// given audio recording (more specifically, its log mel spectrogram), runs forward pass of the encoder
// part of the transformer model and returns the encoded features
//
//   - wctx:      the model
//   - wstate:     the state of the encoder
//   - n_threads:  number of threads to use
//   - mel_offset: offset in the mel spectrogram (i.e. audio offset)
//
static bool whisper_encode_qwen2_internal(
        whisper_context & wctx,
          whisper_state & wstate,
              const int   mel_offset,
              const int   n_threads,
    ggml_abort_callback   abort_callback,
                   void * abort_callback_data) {
    const int64_t t_start_us = ggml_time_us();

    // conv
    {
        auto & sched = wstate.sched_conv.sched;

        ggml_cgraph * gf = whisper_build_graph_conv(wctx, wstate);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        struct ggml_tensor * mel = ggml_graph_get_tensor(gf, "mel");

        // set the input
        {
            const auto & mel_inp = wstate.mel;
            const int n_ctx      = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : wctx.model.hparams.n_audio_ctx;

            assert(mel->type == GGML_TYPE_F32);
            assert(mel_inp.n_mel == wctx.model.hparams.n_mels);

            wstate.inp_mel.resize(ggml_nelements(mel));

            float * dst = wstate.inp_mel.data();
            memset(dst, 0, ggml_nbytes(mel));

            const int i0 = std::min(mel_offset,           mel_inp.n_len);
            const int i1 = std::min(mel_offset + 2*n_ctx, mel_inp.n_len);

            for (int j = 0; j < mel_inp.n_mel; ++j) {
                for (int i = i0; i < i1; ++i) {
                    dst[j*2*n_ctx + (i - i0)] = mel_inp.data[j*mel_inp.n_len + i];
                }
            }

            ggml_backend_tensor_set(mel, wstate.inp_mel.data(), 0, ggml_nelements(mel)*sizeof(float));
        }
        //ggml_graph_print(gf);
        if (!whisper_encode_external(wstate)) {
            if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
                return false;
            }
        } else {
#if defined(WHISPER_USE_COREML)
            whisper_coreml_encode(wstate.ctx_coreml, mel->ne[0], mel->ne[1], (float *) mel->data, (float *) wstate.embd_enc->data);
#elif defined(WHISPER_USE_OPENVINO)
            whisper_openvino_encode(wstate.ctx_openvino, mel, wstate.embd_enc);
#endif
        }
    }

    // encoder
    if (!whisper_encode_external(wstate)) {
        auto & sched = wstate.sched_encode.sched;

        //ggml_backend_sched_set_eval_callback(sched, ggml_debug, nullptr);

        ggml_cgraph * gf = whisper_build_graph_encoder(wctx, wstate);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
    }

    // // cross
    // {
    //     auto & sched = wstate.sched_cross.sched;

    //     ggml_cgraph * gf = whisper_build_graph_cross(wctx, wstate);

    //     if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    //         // should never happen as we pre-allocate the memory
    //         return false;
    //     }

    //     if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
    //         return false;
    //     }
    // }

    wstate.t_encode_us += ggml_time_us() - t_start_us;
    wstate.n_encode++;

    return !(abort_callback && abort_callback(abort_callback_data));
}

int whisper_encoder_output_with_state(
    struct whisper_context * ctx,
          struct whisper_state * state,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {


    if (n_samples > 0) {
        // compute log mel spectrogram
        if (whisper_pcm_to_mel_with_state(ctx, state, samples, n_samples, params.n_threads) != 0) {
            WHISPER_LOG_ERROR("%s: failed to compute log mel spectrogram\n", __func__);
            return -2;
        }
    }
    const int seek_start = params.offset_ms/10;
    const int seek_end = params.duration_ms == 0 ? whisper_n_len_from_state(state) : seek_start + params.duration_ms/10;

    // if length of spectrogram is less than 1.0s (100 frames), then return
    // basically don't process anything that is less than 1.0s
    // see issue #39: https://github.com/ggerganov/whisper.cpp/issues/39
    if (seek_end < seek_start + 100) {
        WHISPER_LOG_WARN("%s: input is too short - %d ms < 1000 ms. consider padding the input audio with silence\n", __func__, (seek_end - seek_start)*10);
        return 0;
    }
    int seek = seek_start;


    if (!whisper_encode_qwen2_internal(*ctx, *state, seek, params.n_threads, params.abort_callback, params.abort_callback_user_data)) {
            WHISPER_LOG_ERROR("%s: failed to encode\n", __func__);
            return -1;
    }
    return 0;

}

int whisper_full(
        struct whisper_context * ctx,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
    return whisper_encoder_output_with_state(ctx, ctx->state, params, samples, n_samples);
}

//  500 -> 00:05.000
// 6000 -> 01:00.000
static std::string to_timestamp(int64_t t, bool comma = false) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

#define SIN_COS_N_COUNT WHISPER_N_FFT
namespace {
struct whisper_global_cache {
    // In FFT, we frequently use sine and cosine operations with the same values.
    // We can use precalculated values to speed up the process.
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];

    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    float hann_window[WHISPER_N_FFT];

    whisper_global_cache() {
        fill_sin_cos_table();
        fill_hann_window(sizeof(hann_window)/sizeof(hann_window[0]), true, hann_window);
    }

    void fill_sin_cos_table() {
        for (int i = 0; i < SIN_COS_N_COUNT; i++) {
            double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
        }
    }

    void fill_hann_window(int length, bool periodic, float * output) {
        int offset = -1;
        if (periodic) {
            offset = 0;
        }
        for (int i = 0; i < length; i++) {
            output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }
} global_cache;
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const float* in, int N, float* out) {
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
            re += in[n]*global_cache.cos_vals[idx]; // cos(t)
            im -= in[n]*global_cache.sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N*2 == 1) {
        dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i]= in[2*i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2*i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = global_cache.cos_vals[idx]; // cos(t)
        float im = -global_cache.sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

static void log_mel_spectrogram_worker_thread(int ith, const float * hann, const std::vector<float> & samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const whisper_filters & filters, whisper_mel & mel) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft = filters.n_fft;
    int i = ith;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    assert(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }
            sum = log10(std::max(sum, 1e-10));
            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool log_mel_spectrogram(
              whisper_state & wstate,
              const float * samples,
              const int   n_samples,
              const int   /*sample_rate*/,
              const int   frame_size,
              const int   frame_step,
              const int   n_mel,
              const int   n_threads,
              const whisper_filters & filters,
              const bool   debug,
              whisper_mel & mel) {
    const int64_t t_start_us = ggml_time_us();

    // Hann window
    WHISPER_ASSERT(frame_size == WHISPER_N_FFT && "Unsupported frame_size");
    const float * hann = global_cache.hann_window;

    // Calculate the length of padding
    int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30;
    int64_t stage_2_pad = frame_size / 2;

    // Initialize a vector and copy data from C array to it.
    std::vector<float> samples_padded;
    samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
    std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

    // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
    std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

    // reflective pad 200 samples at the beginning of audio
    std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());

    mel.n_mel     = n_mel;
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
    // Calculate number of frames + remove the last frame
    mel.n_len     = (samples_padded.size() - frame_size) / frame_step;
    // Calculate semi-padded sample length to ensure compatibility
    mel.n_len_org = 1 + (n_samples + stage_2_pad - frame_size) / frame_step;
    mel.data.resize(mel.n_mel * mel.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] = std::thread(
                    log_mel_spectrogram_worker_thread, iw + 1, hann, samples_padded,
                    n_samples + stage_2_pad, frame_size, frame_step, n_threads,
                    std::cref(filters), std::ref(mel));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, frame_size, frame_step, n_threads, filters, mel);

        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0)/4.0;
    }

    wstate.t_mel_us += ggml_time_us() - t_start_us;

    // Dump log_mel_spectrogram
    if (debug) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < mel.data.size() - 1; i++) {
            outFile << mel.data[i] << ", ";
        }
        outFile << mel.data[mel.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

// split text into tokens
//
// ref: https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L53
//
// Regex (Python):
// r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
//
// Regex (C++):
// R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"
//
static std::vector<whisper_vocab::id> tokenize(const whisper_vocab & vocab, const std::string & text) {
    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;
        std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    // find the longest tokens that form the words:
    std::vector<whisper_vocab::id> tokens;
    for (const auto & word : words) {
        if (word.empty()) continue;

        int i = 0;
        int n = word.size();
        while (i < n) {
            int j = n;
            bool found = false;
            while (j > i) {
                auto sub = word.substr(i, j-i);
                auto it = vocab.token_to_id.find(sub);
                if (it != vocab.token_to_id.end()) {
                    tokens.push_back(it->second);
                    i = j;
                    found = true;
                    break;
                }
                --j;
            }
            if (!found) {
                WHISPER_LOG_ERROR("unknown token\n");
                ++i;
            }
        }
    }

    return tokens;
}

//
// interface implementation
//

#ifdef WHISPER_USE_COREML
// replace .bin with -encoder.mlmodelc
static std::string whisper_get_coreml_path_encoder(std::string path_bin) {
    auto pos = path_bin.rfind('.');
    if (pos != std::string::npos) {
        path_bin = path_bin.substr(0, pos);
    }

    // match "-qx_x"
    pos = path_bin.rfind('-');
    if (pos != std::string::npos) {
        auto sub = path_bin.substr(pos);
        if (sub.size() == 5 && sub[1] == 'q' && sub[3] == '_') {
            path_bin = path_bin.substr(0, pos);
        }
    }

    path_bin += "-encoder.mlmodelc";

    return path_bin;
}
#endif

#ifdef WHISPER_USE_OPENVINO
// replace .bin with-encoder-openvino.xml
static std::string whisper_openvino_get_path_encoder(std::string path_bin) {
    auto pos = path_bin.rfind('.');
    if (pos != std::string::npos) {
        path_bin = path_bin.substr(0, pos);
    }

    path_bin += "-encoder-openvino.xml";

    return path_bin;
}

static std::string whisper_openvino_get_path_cache(std::string path_bin) {
    auto pos = path_bin.rfind('.');
    if (pos != std::string::npos) {
        path_bin = path_bin.substr(0, pos);
    }

    path_bin += "-encoder-openvino-cache";

    return path_bin;
}
#endif

struct whisper_state * whisper_init_state(whisper_context * ctx) {
    whisper_state * state = new whisper_state;

    state->backends = whisper_backend_init(ctx->params);
    if (state->backends.empty()) {
        WHISPER_LOG_ERROR("%s: whisper_backend_init() failed\n", __func__);
        whisper_free_state(state);
        return nullptr;
    }

    //lfr
    // at this point, we don't know yet how many decoders will be used
    // later during decoding, if more decoders are used, we will recreate the KV cache respectively
    // state->kv_self_n_dec = 1;
    // if (!whisper_kv_cache_init(state->kv_self, state->backends[0], ctx->itype,
    //             ctx->model.hparams.n_text_state,
    //             ctx->model.hparams.n_text_layer,
    //             GGML_PAD(ctx->model.hparams.n_text_ctx, 256))) {
    //     WHISPER_LOG_ERROR("%s: whisper_kv_cache_init() failed for self-attention cache\n", __func__);
    //     whisper_free_state(state);
    //     return nullptr;
    // }

    // {
    //     const size_t memory_size = ggml_nbytes(state->kv_self.k) + ggml_nbytes(state->kv_self.v);
    //     WHISPER_LOG_INFO("%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1e6);
    // }

    // if (!whisper_kv_cache_init(state->kv_cross, state->backends[0], ctx->itype,
    //             ctx->model.hparams.n_text_state,
    //             ctx->model.hparams.n_text_layer,
    //             GGML_PAD(ctx->model.hparams.n_audio_ctx, 256))) {
    //     WHISPER_LOG_ERROR("%s: whisper_kv_cache_init() failed for cross-attention cache\n", __func__);
    //     whisper_free_state(state);
    //     return nullptr;
    // }

    // {
    //     const size_t memory_size = ggml_nbytes(state->kv_cross.k) + ggml_nbytes(state->kv_cross.v);
    //     WHISPER_LOG_INFO("%s: kv cross size = %7.2f MB\n", __func__, memory_size / 1e6);
    // }

    // if (!whisper_kv_cache_init(state->kv_pad, state->backends[0], ctx->itype,
    //             ctx->model.hparams.n_audio_state,
    //             1,
    //             GGML_PAD(ctx->model.hparams.n_audio_ctx, 256))) {
    //     WHISPER_LOG_ERROR("%s: whisper_kv_cache_init() failed for self-attention cache\n", __func__);
    //     whisper_free_state(state);
    //     return nullptr;
    // }

    // {
    //     const size_t memory_size = ggml_nbytes(state->kv_pad.k) + ggml_nbytes(state->kv_pad.v);
    //     WHISPER_LOG_INFO("%s: kv pad  size  = %7.2f MB\n", __func__, memory_size / 1e6);
    // }

    // [EXPERIMENTAL] Token-level timestamps with DTW
    if (ctx->params.dtw_token_timestamps) {
        if (!aheads_masks_init(ctx->params, ctx->model.hparams, state->aheads_masks, state->backends[0])) {
            WHISPER_LOG_ERROR("%s: aheads_masks_init() failed for alignment heads masks\n", __func__);
            whisper_free_state(state);
            return nullptr;
        }
        const size_t memory_size = aheads_masks_nbytes(state->aheads_masks);
        WHISPER_LOG_INFO("%s: alignment heads masks size = %ld B\n", __func__, memory_size);
    }

#ifdef WHISPER_USE_COREML
    const auto path_coreml = whisper_get_coreml_path_encoder(ctx->path_model);

    WHISPER_LOG_INFO("%s: loading Core ML model from '%s'\n", __func__, path_coreml.c_str());
    WHISPER_LOG_INFO("%s: first run on a device may take a while ...\n", __func__);

    state->ctx_coreml = whisper_coreml_init(path_coreml.c_str());
    if (!state->ctx_coreml) {
        WHISPER_LOG_ERROR("%s: failed to load Core ML model from '%s'\n", __func__, path_coreml.c_str());
#ifndef WHISPER_COREML_ALLOW_FALLBACK
        whisper_free_state(state);
        return nullptr;
#endif
    } else {
        WHISPER_LOG_INFO("%s: Core ML model loaded\n", __func__);
    }
#endif

    // state->logits.reserve(ctx->vocab.n_vocab * ctx->model.hparams.n_text_ctx);

    // state->batch = whisper_batch_init(ctx->model.hparams.n_text_ctx, WHISPER_MAX_DECODERS);

    // // TAGS: WHISPER_DECODER_INIT
    // state->decoders[0].sequence.tokens.reserve(ctx->model.hparams.n_text_ctx);

    // state->decoders[0].probs.reserve    (ctx->vocab.n_vocab);
    // state->decoders[0].logits.reserve   (ctx->vocab.n_vocab);
    // state->decoders[0].logprobs.reserve (ctx->vocab.n_vocab);
    // state->decoders[0].logits_id.reserve(ctx->model.hparams.n_vocab);

    // state->decoders[0].rng = std::mt19937(0);

    // conv allocator
    {
        bool ok = whisper_sched_graph_init(state->sched_conv, state->backends,
                [&]() {
                    return whisper_build_graph_conv(*ctx, *state);
                });

        if (!ok) {
            WHISPER_LOG_ERROR("%s: failed to init conv allocator\n", __func__);
            whisper_free_state(state);
            return nullptr;
        }

        WHISPER_LOG_INFO("%s: compute buffer (conv)   = %7.2f MB\n", __func__, whisper_sched_size(state->sched_conv) / 1e6);
    }

    // encoder allocator
    if (!whisper_encode_external(*state)) {
        bool ok = whisper_sched_graph_init(state->sched_encode, state->backends,
                [&]() {
                    return whisper_build_graph_encoder(*ctx, *state);
                });

        if (!ok) {
            WHISPER_LOG_ERROR("%s: failed to init encoder allocator\n", __func__);
            whisper_free_state(state);
            return nullptr;
        }

        WHISPER_LOG_INFO("%s: compute buffer (encode) = %7.2f MB\n", __func__, whisper_sched_size(state->sched_encode) / 1e6);
    }

    // // cross allocator
    // {
    //     bool ok = whisper_sched_graph_init(state->sched_cross, state->backends,
    //             [&]() {
    //                 return whisper_build_graph_cross(*ctx, *state);
    //             });

    //     if (!ok) {
    //         WHISPER_LOG_ERROR("%s: failed to init cross allocator\n", __func__);
    //         whisper_free_state(state);
    //         return nullptr;
    //     }

    //     WHISPER_LOG_INFO("%s: compute buffer (cross)  = %7.2f MB\n", __func__, whisper_sched_size(state->sched_cross) / 1e6);
    // }

    // // decoder allocator
    // {
    //     bool ok = whisper_sched_graph_init(state->sched_decode, state->backends,
    //             [&]() {
    //                 const auto & hparams = ctx->model.hparams;

    //                 // TODO: make sure this is the worst-case scenario
    //                 const int n_tokens = hparams.n_text_ctx;
    //                 const int n_past   = 0;

    //                 whisper_batch_prep_legacy(state->batch, nullptr, n_tokens, n_past, 0);

    //                 return whisper_build_graph_decoder(*ctx, *state, state->batch, ctx->params.dtw_token_timestamps, true);
    //             });

    //     if (!ok) {
    //         WHISPER_LOG_ERROR("%s: failed to init decoder allocator\n", __func__);
    //         whisper_free_state(state);
    //         return nullptr;
    //     }

    //     WHISPER_LOG_INFO("%s: compute buffer (decode) = %7.2f MB\n", __func__, whisper_sched_size(state->sched_decode) / 1e6);
    // }

    return state;
}

int whisper_ctx_init_openvino_encoder_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
                    const char * model_path,
                    const char * device,
                    const char * cache_dir) {
#ifndef WHISPER_USE_OPENVINO
    (void)(ctx);
    (void)(state);
    (void)(model_path);
    (void)(device);
    (void)(cache_dir);

    return 1;
#else
    if (!model_path && ctx->path_model.empty()) {
        WHISPER_LOG_ERROR("%s: model_path is nullptr, and ctx has no model_path set.\n", __func__);
        return 1;
    }

    std::string path_encoder;
    if (!model_path) {
        //if model_path is not set, attempt to find it in the same directory as ggml-<model>.bin model
        path_encoder = whisper_openvino_get_path_encoder(ctx->path_model);
    } else {
        path_encoder = model_path;
    }

    std::string path_cache;
    if (!cache_dir) {
        //if cache_dir is not set, set it as a dir residing next to ggml-<model>.bin
        path_cache = whisper_openvino_get_path_cache(ctx->path_model);
    } else {
        path_cache = cache_dir;
    }

    WHISPER_LOG_INFO("%s: loading OpenVINO model from '%s'\n", __func__, path_encoder.c_str());
    WHISPER_LOG_INFO("%s: first run on a device may take a while ...\n", __func__);

    state->ctx_openvino = whisper_openvino_init(path_encoder.c_str(), device, path_cache.c_str());
    if (!state->ctx_openvino) {
        WHISPER_LOG_ERROR("%s: failed to init OpenVINO encoder from '%s'\n", __func__, path_encoder.c_str());
        return 1;
    } else {
        WHISPER_LOG_INFO("%s: OpenVINO model loaded\n", __func__);
    }

    return 0;
#endif
}

int whisper_ctx_init_openvino_encoder(
        struct whisper_context * ctx,
                    const char * model_path,
                    const char * device,
                    const char * cache_dir) {
    return whisper_ctx_init_openvino_encoder_with_state(ctx, ctx->state, model_path, device, cache_dir);
}

struct whisper_context_params whisper_context_default_params() {
    struct whisper_context_params result = {
        /*.use_gpu              =*/ true,
        /*.flash_attn           =*/ false,
        /*.gpu_device           =*/ 0,

        /*.dtw_token_timestamps =*/ false,
        /*.dtw_aheads_preset    =*/ WHISPER_AHEADS_NONE,
        /*.dtw_n_top            =*/ -1,
        /*.dtw_aheads           =*/ {
            /*.n_heads          =*/ 0,
            /*.heads            =*/ NULL,
        },
        /*.dtw_mem_size         =*/ 1024*1024*128,
    };
    return result;
}

struct whisper_context * whisper_init_from_file_with_params_no_state(const char * path_model, struct whisper_context_params params) {
    WHISPER_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);
#ifdef _MSC_VER
    // Convert UTF-8 path to wide string (UTF-16) for Windows, resolving character encoding issues.
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring path_model_wide = converter.from_bytes(path_model);
    auto fin = std::ifstream(path_model_wide, std::ios::binary);
#else
    auto fin = std::ifstream(path_model, std::ios::binary);
#endif
    if (!fin) {
        WHISPER_LOG_ERROR("%s: failed to open '%s'\n", __func__, path_model);
        return nullptr;
    }

    whisper_model_loader loader = {};

    loader.context = &fin;

    loader.read = [](void * ctx, void * output, size_t read_size) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->read((char *)output, read_size);
        return read_size;
    };

    loader.eof = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        return fin->eof();
    };

    loader.close = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->close();
    };

    auto ctx = whisper_init_with_params_no_state(&loader, params);

    if (ctx) {
        ctx->path_model = path_model;
    }

    return ctx;
}

struct whisper_context * whisper_init_from_buffer_with_params_no_state(void * buffer, size_t buffer_size, struct whisper_context_params params) {
    struct buf_context {
        uint8_t* buffer;
        size_t size;
        size_t current_offset;
    };

    buf_context ctx = { reinterpret_cast<uint8_t*>(buffer), buffer_size, 0 };

    WHISPER_LOG_INFO("%s: loading model from buffer\n", __func__);

    whisper_model_loader loader = {};

    loader.context = &ctx;

    loader.read = [](void * ctx, void * output, size_t read_size) {
        buf_context * buf = reinterpret_cast<buf_context *>(ctx);

        size_t size_to_copy = buf->current_offset + read_size < buf->size ? read_size : buf->size - buf->current_offset;

        memcpy(output, buf->buffer + buf->current_offset, size_to_copy);
        buf->current_offset += size_to_copy;

        return size_to_copy;
    };

    loader.eof = [](void * ctx) {
        buf_context * buf = reinterpret_cast<buf_context *>(ctx);

        return buf->current_offset >= buf->size;
    };

    loader.close = [](void * /*ctx*/) { };

    return whisper_init_with_params_no_state(&loader, params);
}

struct whisper_context * whisper_init_with_params_no_state(struct whisper_model_loader * loader, struct whisper_context_params params) {
    ggml_time_init();

    if (params.flash_attn && params.dtw_token_timestamps) {
        WHISPER_LOG_WARN("%s: dtw_token_timestamps is not supported with flash_attn - disabling\n", __func__);
        params.dtw_token_timestamps = false;
    }

    WHISPER_LOG_INFO("%s: use gpu    = %d\n", __func__, params.use_gpu);
    WHISPER_LOG_INFO("%s: flash attn = %d\n", __func__, params.flash_attn);
    WHISPER_LOG_INFO("%s: gpu_device = %d\n", __func__, params.gpu_device);
    WHISPER_LOG_INFO("%s: dtw        = %d\n", __func__, params.dtw_token_timestamps);

    whisper_context * ctx = new whisper_context;
    ctx->params = params;

    if (!whisper_model_load(loader, *ctx)) {
        loader->close(loader->context);
        WHISPER_LOG_ERROR("%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }

    loader->close(loader->context);

    return ctx;
}

struct whisper_context * whisper_init_from_file_with_params(const char * path_model, struct whisper_context_params params) {
    whisper_context * ctx = whisper_init_from_file_with_params_no_state(path_model, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = whisper_init_state(ctx);
    if (!ctx->state) {
        whisper_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct whisper_context * whisper_init_from_buffer_with_params(void * buffer, size_t buffer_size, struct whisper_context_params params) {
    whisper_context * ctx = whisper_init_from_buffer_with_params_no_state(buffer, buffer_size, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = whisper_init_state(ctx);
    if (!ctx->state) {
        whisper_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct whisper_context * whisper_init_with_params(struct whisper_model_loader * loader, struct whisper_context_params params) {
    whisper_context * ctx = whisper_init_with_params_no_state(loader, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = whisper_init_state(ctx);
    if (!ctx->state) {
        whisper_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct whisper_context * whisper_init_from_file(const char * path_model) {
    return whisper_init_from_file_with_params(path_model, whisper_context_default_params());
}

struct whisper_context * whisper_init_from_buffer(void * buffer, size_t buffer_size) {
    return whisper_init_from_buffer_with_params(buffer, buffer_size, whisper_context_default_params());
}

struct whisper_context * whisper_init(struct whisper_model_loader * loader) {
    return whisper_init_with_params(loader, whisper_context_default_params());
}

struct whisper_context * whisper_init_from_file_no_state(const char * path_model) {
    return whisper_init_from_file_with_params_no_state(path_model, whisper_context_default_params());
}

struct whisper_context * whisper_init_from_buffer_no_state(void * buffer, size_t buffer_size) {
    return whisper_init_from_buffer_with_params_no_state(buffer, buffer_size, whisper_context_default_params());
}

struct whisper_context * whisper_init_no_state(struct whisper_model_loader * loader) {
    return whisper_init_with_params_no_state(loader, whisper_context_default_params());
}

void whisper_free_state(struct whisper_state * state) {
    if (state) {
        whisper_kv_cache_free(state->kv_pad);

#ifdef WHISPER_USE_COREML
        if (state->ctx_coreml != nullptr) {
            whisper_coreml_free(state->ctx_coreml);
            state->ctx_coreml = nullptr;
        }
#endif

#ifdef WHISPER_USE_OPENVINO
        if (state->ctx_openvino != nullptr) {
            whisper_openvino_free(state->ctx_openvino);
            state->ctx_openvino = nullptr;
        }
#endif

        //whisper_batch_free(state->batch);

        ggml_backend_sched_free(state->sched_conv.sched);
        ggml_backend_sched_free(state->sched_encode.sched);
        ggml_backend_sched_free(state->sched_cross.sched);
        ggml_backend_sched_free(state->sched_decode.sched);

        for (auto & backend : state->backends) {
            ggml_backend_free(backend);
        }

        // [EXPERIMENTAL] Token-level timestamps with DTW
        aheads_masks_free(state->aheads_masks);

        delete state;
    }
}

void whisper_free(struct whisper_context * ctx) {
    if (ctx) {
        ggml_free(ctx->model.ctx);

        ggml_backend_buffer_free(ctx->model.buffer);

        whisper_free_state(ctx->state);

        delete ctx;
    }
}

void whisper_free_context_params(struct whisper_context_params * params) {
    if (params) {
        delete params;
    }
}

void whisper_free_params(struct whisper_full_params * params) {
    if (params) {
        delete params;
    }
}

int whisper_pcm_to_mel_with_state(struct whisper_context * ctx, struct whisper_state * state, const float * samples, int n_samples, int n_threads) {
    if (!log_mel_spectrogram(*state, samples, n_samples, WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, ctx->model.filters.n_mel, n_threads, ctx->model.filters, false, state->mel)) {
        WHISPER_LOG_ERROR("%s: failed to compute mel spectrogram\n", __func__);
        return -1;
    }

    return 0;
}

int whisper_pcm_to_mel(struct whisper_context * ctx, const float * samples, int n_samples, int n_threads) {
    return whisper_pcm_to_mel_with_state(ctx, ctx->state, samples, n_samples, n_threads);
}

int whisper_set_mel_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
                   const float * data,
                           int   n_len,
                           int   n_mel) {
    if (n_mel != ctx->model.filters.n_mel) {
        WHISPER_LOG_ERROR("%s: invalid number of mel bands: %d (expected %d)\n", __func__, n_mel, ctx->model.filters.n_mel);
        return -1;
    }

    state->mel.n_len     = n_len;
    state->mel.n_len_org = n_len;
    state->mel.n_mel     = n_mel;

    state->mel.data.resize(n_len*n_mel);
    memcpy(state->mel.data.data(), data, n_len*n_mel*sizeof(float));

    return 0;
}

int whisper_set_mel(
        struct whisper_context * ctx,
        const float * data,
        int n_len,
        int n_mel) {
    return whisper_set_mel_with_state(ctx, ctx->state, data, n_len, n_mel);
}


int whisper_tokenize(struct whisper_context * ctx, const char * text, whisper_token * tokens, int n_max_tokens) {
    const auto res = tokenize(ctx->vocab, text);

    if (n_max_tokens < (int) res.size()) {
        WHISPER_LOG_ERROR("%s: too many resulting tokens: %d (max %d)\n", __func__, (int) res.size(), n_max_tokens);
        return -(int) res.size();
    }

    for (int i = 0; i < (int) res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int whisper_token_count(struct whisper_context * ctx, const char * text) {
    return -whisper_tokenize(ctx, text, NULL, 0);
}

int whisper_lang_max_id(void) {
    auto max_id = 0;
    for (const auto & kv : g_lang) {
        max_id = std::max(max_id, kv.second.first);
    }

    return max_id;
}

int whisper_lang_id(const char * lang) {
    if (!g_lang.count(lang)) {
        for (const auto & kv : g_lang) {
            if (kv.second.second == lang) {
                return kv.second.first;
            }
        }

        WHISPER_LOG_ERROR("%s: unknown language '%s'\n", __func__, lang);
        return -1;
    }
    return g_lang.at(lang).first;
}

const char * whisper_lang_str(int id) {
    for (const auto & kv : g_lang) {
        if (kv.second.first == id) {
            return kv.first.c_str();
        }
    }

    WHISPER_LOG_ERROR("%s: unknown language id %d\n", __func__, id);
    return nullptr;
}

const char * whisper_lang_str_full(int id) {
   for (const auto & kv : g_lang) {
        if (kv.second.first == id) {
            return kv.second.second.c_str();
        }
    }

    WHISPER_LOG_ERROR("%s: unknown language id %d\n", __func__, id);
    return nullptr;
}

int whisper_model_n_vocab(struct whisper_context * ctx) {
    return ctx->model.hparams.n_vocab;
}

int whisper_model_n_audio_ctx(struct whisper_context * ctx) {
    return ctx->model.hparams.n_audio_ctx;
}

int whisper_model_n_audio_state(struct whisper_context * ctx) {
    return ctx->model.hparams.n_audio_state;
}

int whisper_model_n_audio_head(struct whisper_context * ctx) {
    return ctx->model.hparams.n_audio_head;
}

int whisper_model_n_audio_layer(struct whisper_context * ctx) {
    return ctx->model.hparams.n_audio_layer;
}

int whisper_model_n_text_ctx(struct whisper_context * ctx) {
    return ctx->model.hparams.n_text_ctx;
}

int whisper_model_n_text_state(struct whisper_context * ctx) {
    return ctx->model.hparams.n_text_state;
}

int whisper_model_n_text_head(struct whisper_context * ctx) {
    return ctx->model.hparams.n_text_head;
}

int whisper_model_n_text_layer(struct whisper_context * ctx) {
    return ctx->model.hparams.n_text_layer;
}

int whisper_model_n_mels(struct whisper_context * ctx) {
    return ctx->model.hparams.n_mels;
}

int whisper_model_ftype(struct whisper_context * ctx) {
    return ctx->model.hparams.ftype;
}

int whisper_model_type(struct whisper_context * ctx) {
    return ctx->model.type;
}

const char *whisper_model_type_readable(struct whisper_context * ctx) {
    switch (ctx->model.type) {
    case e_model::MODEL_TINY:
        return "tiny";
    case e_model::MODEL_BASE:
        return "base";
    case e_model::MODEL_SMALL:
        return "small";
    case e_model::MODEL_MEDIUM:
        return "medium";
    case e_model::MODEL_LARGE:
        return "large";
    default:
        return "unknown";
    }
}

int whisper_n_len_from_state(struct whisper_state * state) {
    return state->mel.n_len_org;
}

int whisper_n_len(struct whisper_context * ctx) {
    return ctx->state->mel.n_len_org;
}

int whisper_n_vocab(struct whisper_context * ctx) {
    return ctx->vocab.n_vocab;
}

int whisper_n_text_ctx(struct whisper_context * ctx) {
    return ctx->model.hparams.n_text_ctx;
}

int whisper_n_audio_ctx(struct whisper_context * ctx) {
    return ctx->model.hparams.n_audio_ctx;
}

int whisper_is_multilingual(struct whisper_context * ctx) {
    return ctx->vocab.is_multilingual() ? 1 : 0;
}

float * whisper_get_logits(struct whisper_context * ctx) {
    return ctx->state->logits.data();
}

float * whisper_get_logits_from_state(struct whisper_state * state) {
    return state->logits.data();
}

const char * whisper_token_to_str(struct whisper_context * ctx, whisper_token token) {
    return ctx->vocab.id_to_token.at(token).c_str();
}

whisper_token whisper_token_eot(struct whisper_context * ctx) {
    return ctx->vocab.token_eot;
}

whisper_token whisper_token_sot(struct whisper_context * ctx) {
    return ctx->vocab.token_sot;
}

whisper_token whisper_token_solm(struct whisper_context * ctx) {
    return ctx->vocab.token_solm;
}

whisper_token whisper_token_prev(struct whisper_context * ctx) {
    return ctx->vocab.token_prev;
}

whisper_token whisper_token_nosp(struct whisper_context * ctx) {
    return ctx->vocab.token_nosp;
}

whisper_token whisper_token_not(struct whisper_context * ctx) {
    return ctx->vocab.token_not;
}

whisper_token whisper_token_beg(struct whisper_context * ctx) {
    return ctx->vocab.token_beg;
}

whisper_token whisper_token_lang(struct whisper_context * ctx, int lang_id) {
    return whisper_token_sot(ctx) + 1 + lang_id;
}

whisper_token whisper_token_translate(struct whisper_context * ctx) {
    return ctx->vocab.token_translate;
}

whisper_token whisper_token_transcribe(struct whisper_context * ctx) {
    return ctx->vocab.token_transcribe;
}

void whisper_print_timings(struct whisper_context * ctx) {
    const int64_t t_end_us = ggml_time_us();

    WHISPER_LOG_INFO("\n");
    WHISPER_LOG_INFO("%s:     load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0f);
    if (ctx->state != nullptr) {

        const int32_t n_sample = std::max(1, ctx->state->n_sample);
        const int32_t n_encode = std::max(1, ctx->state->n_encode);
        const int32_t n_decode = std::max(1, ctx->state->n_decode);
        const int32_t n_batchd = std::max(1, ctx->state->n_batchd);
        const int32_t n_prompt = std::max(1, ctx->state->n_prompt);

        WHISPER_LOG_INFO("%s:     fallbacks = %3d p / %3d h\n", __func__, ctx->state->n_fail_p, ctx->state->n_fail_h);
        WHISPER_LOG_INFO("%s:      mel time = %8.2f ms\n", __func__, ctx->state->t_mel_us / 1000.0f);
        WHISPER_LOG_INFO("%s:   sample time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_sample_us, n_sample, 1e-3f * ctx->state->t_sample_us / n_sample);
        WHISPER_LOG_INFO("%s:   encode time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_encode_us, n_encode, 1e-3f * ctx->state->t_encode_us / n_encode);
        WHISPER_LOG_INFO("%s:   decode time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_decode_us, n_decode, 1e-3f * ctx->state->t_decode_us / n_decode);
        WHISPER_LOG_INFO("%s:   batchd time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_batchd_us, n_batchd, 1e-3f * ctx->state->t_batchd_us / n_batchd);
        WHISPER_LOG_INFO("%s:   prompt time = %8.2f ms / %5d runs (%8.2f ms per run)\n", __func__, 1e-3f * ctx->state->t_prompt_us, n_prompt, 1e-3f * ctx->state->t_prompt_us / n_prompt);
    }
    WHISPER_LOG_INFO("%s:    total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us)/1000.0f);
}

void whisper_reset_timings(struct whisper_context * ctx) {
    ctx->t_start_us = ggml_time_us();
    if (ctx->state != nullptr) {
        ctx->state->t_mel_us = 0;
        ctx->state->t_sample_us = 0;
        ctx->state->t_encode_us = 0;
        ctx->state->t_decode_us = 0;
        ctx->state->t_batchd_us = 0;
        ctx->state->t_prompt_us = 0;
        ctx->state->n_sample = 0;
        ctx->state->n_encode = 0;
        ctx->state->n_decode = 0;
        ctx->state->n_batchd = 0;
        ctx->state->n_prompt = 0;
    }
}

static int whisper_has_coreml(void) {
#ifdef WHISPER_USE_COREML
    return 1;
#else
    return 0;
#endif
}

static int whisper_has_openvino(void) {
#ifdef WHISPER_USE_OPENVINO
    return 1;
#else
    return 0;
#endif
}

const char * whisper_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "METAL = "     + std::to_string(ggml_cpu_has_metal())     + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "SSSE3 = "     + std::to_string(ggml_cpu_has_ssse3())     + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";
    s += "CUDA = "      + std::to_string(ggml_cpu_has_cuda())      + " | ";
    s += "COREML = "    + std::to_string(whisper_has_coreml())     + " | ";
    s += "OPENVINO = "  + std::to_string(whisper_has_openvino())   + " | ";
    s += "CANN = "      + std::to_string(ggml_cpu_has_cann())             ;
    return s.c_str();
}

//////////////////////////////////
// Grammar - ported from llama.cpp
//////////////////////////////////

// Decodes a UTF-8 string which may end in an incomplete sequence. Adds a terminating 0 for use as
// pointer. If an invalid sequence is encountered, returns `whisper_partial_utf8.n_remain == -1`.
static std::pair<std::vector<uint32_t>, whisper_partial_utf8> decode_utf8(
        const char         * src,
        whisper_partial_utf8   partial_start) {
    static const int      lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    const char          * pos      = src;
    std::vector<uint32_t> code_points;
    uint32_t              value    = partial_start.value;
    int                   n_remain = partial_start.n_remain;

    // continue previous decode, if applicable
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            // invalid sequence, abort
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), whisper_partial_utf8{ 0, -1 });
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }

    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // decode any subsequent utf-8 sequences, which may end in an incomplete one
    while (*pos != 0) {
        uint8_t  first_byte = static_cast<uint8_t>(*pos);
        uint8_t  highbits   = first_byte >> 4;
                 n_remain   = lookup[highbits] - 1;

        if (n_remain < 0) {
            // invalid sequence, abort
            code_points.clear();
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), whisper_partial_utf8{ 0, n_remain });
        }

        uint8_t  mask       = (1 << (7 - n_remain)) - 1;
                 value      = first_byte & mask;
        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }
    code_points.push_back(0);

    return std::make_pair(std::move(code_points), whisper_partial_utf8{ value, n_remain });
}

////////////////////////////////////////////////////////////////////////////

struct whisper_context_params * whisper_context_default_params_by_ref(void) {
    struct whisper_context_params params = whisper_context_default_params();

    struct whisper_context_params* result = new whisper_context_params();
    *result = params;
    return result;
}

// forward declarations
static std::vector<float> get_signal_energy(const float * signal, int n_samples, int n_samples_per_half_window);
static void whisper_exp_compute_token_level_timestamps(
        struct whisper_context & ctx,
          struct whisper_state & state,
                           int   i_segment,
                         float   thold_pt,
                         float   thold_ptsum);

static inline bool should_split_on_word(const char * txt, bool split_on_word) {
    if (!split_on_word) return true;

    return txt[0] == ' ';
}

//
// Temporary interface needed for exposing ggml interface
// Will be removed in the future when ggml becomes a separate library
//

WHISPER_API int whisper_bench_memcpy(int n_threads) {
    fputs(whisper_bench_memcpy_str(n_threads), stderr);
    return 0;
}

WHISPER_API const char * whisper_bench_memcpy_str(int n_threads) {
    static std::string s;
    s = "";
    char strbuf[256];

    ggml_time_init();

    size_t n    = 20;
    size_t arr  = n_threads > 0 ? 1024llu : n_threads; // trick to avoid compiler optimizations

    // 1GB array
    const size_t size = arr*1e6;

    double sum  = 0.0;

    // heat-up
    {
        char * src = (char *) malloc(size);
        char * dst = (char *) malloc(size);

        for (size_t i = 0; i < size; i++) src[i] = i;

        memcpy(dst, src, size); // heat-up

        double tsum = 0.0;

        for (size_t i = 0; i < n; i++) {
            const int64_t t0 = ggml_time_us();

            memcpy(dst, src, size);

            const int64_t t1 = ggml_time_us();

            tsum += (t1 - t0)*1e-6;

            src[rand() % size] = rand() % 256;
        }

        snprintf(strbuf, sizeof(strbuf), "memcpy: %7.2f GB/s (heat-up)\n", (double) (n*size)/(tsum*1e9));
        s += strbuf;

        // needed to prevent the compiler from optimizing the memcpy away
        {
            for (size_t i = 0; i < size; i++) sum += dst[i];
        }

        free(src);
        free(dst);
    }

    // single-thread
    {
        char * src = (char *) malloc(size);
        char * dst = (char *) malloc(size);

        for (size_t i = 0; i < size; i++) src[i] = i;

        memcpy(dst, src, size); // heat-up

        double tsum = 0.0;

        for (size_t i = 0; i < n; i++) {
            const int64_t t0 = ggml_time_us();

            memcpy(dst, src, size);

            const int64_t t1 = ggml_time_us();

            tsum += (t1 - t0)*1e-6;

            src[rand() % size] = rand() % 256;
        }

        snprintf(strbuf, sizeof(strbuf), "memcpy: %7.2f GB/s ( 1 thread)\n", (double) (n*size)/(tsum*1e9));
        s += strbuf;

        // needed to prevent the compiler from optimizing the memcpy away
        {
            for (size_t i = 0; i < size; i++) sum += dst[i];
        }

        free(src);
        free(dst);
    }

    // multi-thread

    for (int32_t k = 1; k <= n_threads; k++) {
        char * src = (char *) malloc(size);
        char * dst = (char *) malloc(size);

        for (size_t i = 0; i < size; i++) src[i] = i;

        memcpy(dst, src, size); // heat-up

        double tsum = 0.0;

        auto helper = [&](int th) {
            const int64_t i0 = (th + 0)*size/k;
            const int64_t i1 = (th + 1)*size/k;

            for (size_t i = 0; i < n; i++) {
                memcpy(dst + i0, src + i0, i1 - i0);

                src[i0 + rand() % (i1 - i0)] = rand() % 256;
            };
        };

        const int64_t t0 = ggml_time_us();

        std::vector<std::thread> threads(k - 1);
        for (int32_t th = 0; th < k - 1; ++th) {
            threads[th] = std::thread(helper, th);
        }

        helper(k - 1);

        for (int32_t th = 0; th < k - 1; ++th) {
            threads[th].join();
        }

        const int64_t t1 = ggml_time_us();

        tsum += (t1 - t0)*1e-6;

        snprintf(strbuf, sizeof(strbuf), "memcpy: %7.2f GB/s (%2d thread)\n", (double) (n*size)/(tsum*1e9), k);
        s += strbuf;

        // needed to prevent the compiler from optimizing the memcpy away
        {
            for (size_t i = 0; i < size; i++) sum += dst[i];
        }

        free(src);
        free(dst);
    }

    snprintf(strbuf, sizeof(strbuf), "sum:    %f\n", sum);
    s += strbuf;

    return s.c_str();
}

WHISPER_API int whisper_bench_ggml_mul_mat(int n_threads) {
    fputs(whisper_bench_ggml_mul_mat_str(n_threads), stderr);
    return 0;
}

WHISPER_API const char * whisper_bench_ggml_mul_mat_str(int n_threads) {
    static std::string s;
    s = "";
    char strbuf[256];

    ggml_time_init();

    const int n_max = 128;

    const std::vector<size_t> sizes = {
        64, 128, 256, 512, 1024, 2048, 4096,
    };

    const size_t N_max = sizes.back();

    // a: N*N*sizeof(float)
    // b: N*N*sizeof(float)
    // c: N*N*sizeof(float)
    // when F16 is used, there is an extra work buffer of size N*N*sizeof(float)
    std::vector<uint8_t> buf(3llu*N_max*N_max*sizeof(float) + 3*ggml_tensor_overhead() + ggml_graph_overhead());
    std::vector<uint8_t> work;

    // put a bunch of random data in the buffer
    for (size_t i = 0; i < buf.size(); i++) buf[i] = i;

    for (int j = 0; j < (int) sizes.size(); j++) {
        int n_q4_0 = 0;
        int n_q4_1 = 0;
        int n_q5_0 = 0;
        int n_q5_1 = 0;
        int n_q8_0 = 0;
        int n_fp16 = 0;
        int n_fp32 = 0;

        // GFLOPS/s
        double s_q4_0 = 0.0;
        double s_q4_1 = 0.0;
        double s_q5_0 = 0.0;
        double s_q5_1 = 0.0;
        double s_q8_0 = 0.0;
        double s_fp16 = 0.0;
        double s_fp32 = 0.0;

        const size_t N = sizes[j];

        for (int k = 0; k < 7; ++k) {
            const ggml_type wtype =
                k == 0 ? GGML_TYPE_Q4_0 :
                k == 1 ? GGML_TYPE_Q4_1 :
                k == 2 ? GGML_TYPE_Q5_0 :
                k == 3 ? GGML_TYPE_Q5_1 :
                k == 4 ? GGML_TYPE_Q8_0 :
                k == 5 ? GGML_TYPE_F16  : GGML_TYPE_F32;

            double & s = k == 0 ? s_q4_0 : k == 1 ? s_q4_1 : k == 2 ? s_q5_0 : k == 3 ? s_q5_1 : k == 4 ? s_q8_0 : k == 5 ? s_fp16 : /*k == 6*/ s_fp32;
            int    & n = k == 0 ? n_q4_0 : k == 1 ? n_q4_1 : k == 2 ? n_q5_0 : k == 3 ? n_q5_1 : k == 4 ? n_q8_0 : k == 5 ? n_fp16 : /*k == 6*/ n_fp32;

            struct ggml_init_params gparams = {
                /*.mem_size   =*/ buf.size(),
                /*.mem_buffer =*/ buf.data(),
                /*.no_alloc   =*/ false,
            };

            struct ggml_context * ctx0 = ggml_init(gparams);

            struct ggml_tensor * a = ggml_new_tensor_2d(ctx0, wtype,         N, N);
            struct ggml_tensor * b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N, N);

            struct ggml_tensor * c = ggml_mul_mat(ctx0, a, b);

            struct ggml_cgraph * gf = ggml_new_graph(ctx0);

            ggml_build_forward_expand(gf, c);

            double tsum = 0.0;

            // heat-up
            ggml_graph_compute_helper(gf, work, n_threads, nullptr, nullptr);

            for (int i = 0; i < n_max; ++i) {
                const int64_t t0 = ggml_time_us();

                ggml_graph_compute_helper(gf, work, n_threads, nullptr, nullptr);

                const int64_t t1 = ggml_time_us();

                tsum += (t1 - t0)*1e-6;
                n++;

                if (tsum > 1.0 && n >= 3) {
                    break;
                }
            }

            ggml_free(ctx0);

            s = ((2.0*N*N*N*n)/tsum)*1e-9;
        }

        // Q4_0 | Q4_1
        snprintf(strbuf, sizeof(strbuf), "%4zu x %4zu: Q4_0 %7.1f GFLOPS (%3d runs) | Q4_1 %7.1f GFLOPS (%3d runs)\n",
                N, N, s_q4_0, n_q4_0, s_q4_1, n_q4_1);
        s += strbuf;

        // Q5_0 | Q5_1 | Q8_0
        snprintf(strbuf, sizeof(strbuf), "%4zu x %4zu: Q5_0 %7.1f GFLOPS (%3d runs) | Q5_1 %7.1f GFLOPS (%3d runs) | Q8_0 %7.1f GFLOPS (%3d runs)\n",
                N, N, s_q5_0, n_q5_0, s_q5_1, n_q5_1, s_q8_0, n_q8_0);
        s += strbuf;

        // F16 | F32
        snprintf(strbuf, sizeof(strbuf), "%4zu x %4zu: F16  %7.1f GFLOPS (%3d runs) | F32  %7.1f GFLOPS (%3d runs)\n",
                N, N, s_fp16, n_fp16, s_fp32, n_fp32);
        s += strbuf;
    }

    return s.c_str();
}

// =================================================================================================

// =================================================================================================

//
// Experimental stuff below
//
// Not sure if these should be part of the library at all, because the quality of the results is not
// guaranteed. Might get removed at some point unless a robust algorithm implementation is found
//

// =================================================================================================

//
// token-level timestamps
//

static int timestamp_to_sample(int64_t t, int n_samples) {
    return std::max(0, std::min((int) n_samples - 1, (int) ((t*WHISPER_SAMPLE_RATE)/100)));
}

static int64_t sample_to_timestamp(int i_sample) {
    return (100ll*i_sample)/WHISPER_SAMPLE_RATE;
}

// a cost-function / heuristic that is high for text that takes longer to pronounce
// obviously, can be improved
static float voice_length(const std::string & text) {
    float res = 0.0f;

    for (char c : text) {
        if (c == ' ') {
            res += 0.01f;
        } else if (c == ',') {
            res += 2.00f;
        } else if (c == '.') {
            res += 3.00f;
        } else if (c == '!') {
            res += 3.00f;
        } else if (c == '?') {
            res += 3.00f;
        } else if (c >= '0' && c <= '9') {
            res += 3.00f;
        } else {
            res += 1.00f;
        }
    }

    return res;
}

// average the fabs of the signal
static std::vector<float> get_signal_energy(const float * signal, int n_samples, int n_samples_per_half_window) {
    const int hw = n_samples_per_half_window;

    std::vector<float> result(n_samples);

    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int j = -hw; j <= hw; j++) {
            if (i + j >= 0 && i + j < n_samples) {
                sum += fabs(signal[i + j]);
            }
        }
        result[i] = sum/(2*hw + 1);
    }

    return result;
}

//
// token level timestamps - dtw version
//

// n_text_layer -> total text layers on model
// n_head -> total heads per text layer on model
static std::vector<uint32_t> get_alignment_heads_by_layer(const whisper_context_params & cparams, int il, int n_text_layer, int n_head) {
    std::vector<uint32_t> ret;
    if (cparams.dtw_aheads_preset == WHISPER_AHEADS_NONE) {
        return ret;
    } else if (cparams.dtw_aheads_preset == WHISPER_AHEADS_N_TOP_MOST) {
        if (il >= n_text_layer - cparams.dtw_n_top) {
            for (int32_t i = 0; i < n_head; ++i) {
                ret.push_back(i);
            }
        }
    } else {
        const auto aheads = cparams.dtw_aheads_preset == WHISPER_AHEADS_CUSTOM ? cparams.dtw_aheads : g_aheads.at(cparams.dtw_aheads_preset);
        for (size_t i = 0; i < aheads.n_heads; ++i) {
            if (aheads.heads[i].n_text_layer == il) {
                ret.push_back(aheads.heads[i].n_head);
            }
        }
    }
    return ret;
}

// dtw + backtrace to return found path
// based on
// https://github.com/openai/whisper/blob/main/whisper/timing.py#L83
static ggml_tensor * dtw_and_backtrace(ggml_context * ctx, ggml_tensor * x) {
    WHISPER_ASSERT(ggml_n_dims(x) == 2);

    int64_t N = x->ne[0];
    int64_t M = x->ne[1];
    struct ggml_tensor * cost = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N + 1, M + 1);
    struct ggml_tensor * trace = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N + 1, M + 1);

    cost = ggml_set_f32(cost, INFINITY);
    trace = ggml_set_f32(trace, -1);
    ggml_set_f32_nd(cost, 0, 0, 0, 0, 0.0);

    // dtw
    // supposedly can be optmized by computing diagonals in parallel ?
    // Not sure it is worth it since x will be GENERATED_TOKENS*1500 size at most.
    for (int64_t j = 1; j < M + 1; ++j) {
        for (int64_t i = 1; i < N + 1; ++i) {
            float c0 = ggml_get_f32_nd(cost, i - 1, j - 1, 0, 0);
            float c1 = ggml_get_f32_nd(cost, i - 1, j, 0, 0);
            float c2 = ggml_get_f32_nd(cost, i, j - 1, 0, 0);

            float c;
            int32_t t;
            if (c0 < c1 && c0 < c2) {
                c = c0;
                t = 0;
            } else if (c1 < c0 && c1 < c2) {
                c = c1;
                t = 1;
            } else {
                c = c2;
                t = 2;
            }

            c = ggml_get_f32_nd(x, i - 1, j - 1, 0, 0) + c;
            ggml_set_f32_nd(cost, i, j, 0, 0, c);
            ggml_set_i32_nd(trace, i, j, 0, 0, t);
        }
    }

    // Backtrace
    const int64_t BT_MAX_ROWS = N + M - 1;
    struct ggml_tensor * bt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, BT_MAX_ROWS, 2);
    // trace[0, :] = 2;
    for (int64_t i = 0; i < M + 1; ++i)
        ggml_set_i32_nd(trace, 0, i, 0, 0, 2);
    //trace[:, 0] = 1;
    for (int64_t i = 0; i < N + 1; ++i)
        ggml_set_i32_nd(trace, i, 0, 0, 0, 1);
    int bt_row_idx = BT_MAX_ROWS - 1;
    int64_t i = N;
    int64_t j = M;
    while (i > 0 || j > 0) {
        ggml_set_i32_nd(bt, bt_row_idx, 0, 0, 0, i - 1);
        ggml_set_i32_nd(bt, bt_row_idx, 1, 0, 0, j - 1);
        --bt_row_idx;

        int32_t t = ggml_get_i32_nd(trace, i, j, 0, 0);
        if (t == 0) {
            --i;
            --j;
        } else if (t == 1) {
            --i;
        } else if (t == 2) {
            --j;
        } else {
            WHISPER_ASSERT(0);
        }
    }

    // FIXME: manual clip/transpose might not be the most efficient way? (e.g. use ggml funcs)
    // Clip + transpose
    // This might not be entirely necessary for our case, but leaving it for now so output matrix
    // is identical to dtw on openAI timing.py
    const int64_t result_n_cols = BT_MAX_ROWS-bt_row_idx-1;
    ggml_tensor * r = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, result_n_cols);
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < result_n_cols; ++j) {
            int32_t v = ggml_get_i32_nd(bt, j+bt_row_idx+1, i, 0, 0);
            ggml_set_i32_nd(r, i, j, 0, 0, v);
        }
    }

    return r;
}

struct median_filter_user_data {
    int filter_width;
};

static void median_filter(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int /*nth*/, void * userdata) {
    if (ith != 0) {
        return;
    }
    int filter_width = ((median_filter_user_data *) userdata)->filter_width;
    WHISPER_ASSERT(filter_width < a->ne[2]);
    WHISPER_ASSERT(filter_width % 2);
    WHISPER_ASSERT(ggml_n_dims(a) == 3);
    WHISPER_ASSERT(a->type == GGML_TYPE_F32);

    std::vector<float> filter;
    filter.reserve(filter_width);
    for (int64_t i = 0; i < a->ne[0]; ++i) {
        for (int64_t j = 0; j < a->ne[1]; ++j) {
            for (int64_t k = 0; k < a->ne[2]; ++k) {
                for (int64_t off = -filter_width/2; off <= filter_width/2; ++off) {
                    // "reflect" padding
                    int64_t idx = k + off;
                    if (idx < 0) {
                        idx = -idx;
                    } else if (idx >= a->ne[2]) {
                        idx = 2*(a->ne[2] - 1) - idx;
                    }

                    filter.push_back(ggml_get_f32_nd(a, i, j, idx, 0));
                }
                std::sort(filter.begin(), filter.end());
                const float v = filter[filter.size()/2];
                ggml_set_f32_nd(dst, i, j, k, 0, v);
                filter.clear();
            }
        }
    }
}

void whisper_log_set(ggml_log_callback log_callback, void * user_data) {
    g_state.log_callback = log_callback ? log_callback : whisper_log_callback_default;
    g_state.log_callback_user_data = user_data;
}

void whisper_print_emb_enc(struct whisper_context * ctx) {
    //data transfer GPU->CPU
    std::vector<float> temp_vec;  //just for test
    temp_vec.resize(20);

    ggml_backend_tensor_get(ctx->state->embd_enc, temp_vec.data(), 0, sizeof(float)*20);

    for (int i =0;i < 20; ++i) {
        printf(" %.3f", temp_vec[i]);
    }
    printf("\n");
    
}


GGML_ATTRIBUTE_FORMAT(2, 3)
static void whisper_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    int len = vsnprintf(buffer, 1024, format, args);
    if (len < 1024) {
        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args);
        buffer2[len] = 0;
        g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args);
}

static void whisper_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct whisper_full_params whisper_full_default_params() {
    struct whisper_full_params result = {
        /*.n_threads         =*/ std::min(4, (int32_t) std::thread::hardware_concurrency()),
        /*.n_max_text_ctx    =*/ 16384,
        /*.offset_ms         =*/ 0,
        /*.duration_ms       =*/ 0,

        /*.translate         =*/ false,
        /*.no_context        =*/ true,
        /*.no_timestamps     =*/ false,
        /*.single_segment    =*/ false,
        /*.print_special     =*/ false,
        /*.print_progress    =*/ true,
        /*.print_realtime    =*/ false,
        /*.print_timestamps  =*/ true,

        /*.token_timestamps  =*/ false,
        /*.thold_pt          =*/ 0.01f,
        /*.thold_ptsum       =*/ 0.01f,
        /*.max_len           =*/ 0,
        /*.split_on_word     =*/ false,
        /*.max_tokens        =*/ 0,

        /*.debug_mode        =*/ false,
        /*.audio_ctx         =*/ 0,

        /*.tdrz_enable       =*/ false,

        /* suppress_regex    =*/ nullptr,

        /*.initial_prompt    =*/ nullptr,
        /*.prompt_tokens     =*/ nullptr,
        /*.prompt_n_tokens   =*/ 0,

        /*.language          =*/ "en",
        /*.detect_language   =*/ false,

        /*.suppress_blank    =*/ true,
        /*.suppress_non_speech_tokens =*/ false,

        /*.temperature       =*/  0.0f,
        /*.max_initial_ts    =*/  1.0f,
        /*.length_penalty    =*/ -1.0f,

        /*.temperature_inc   =*/  0.2f,
        /*.entropy_thold     =*/  2.4f,
        /*.logprob_thold     =*/ -1.0f,
        /*.no_speech_thold   =*/  0.6f,


        /*.new_segment_callback           =*/ nullptr,
        /*.new_segment_callback_user_data =*/ nullptr,

        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,

        /*.encoder_begin_callback           =*/ nullptr,
        /*.encoder_begin_callback_user_data =*/ nullptr,

        /*.abort_callback                   =*/ nullptr,
        /*.abort_callback_user_data         =*/ nullptr,

        /*.i_start_rule    =*/ 0,
    };
}
