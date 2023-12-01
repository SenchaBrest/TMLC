#include "onnxruntime/include/onnxruntime_cxx_api.h"
#include "tglang.h"

enum TglangLanguage tglang_detect_programming_language(const char *text) {
    Ort::Env env;
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    unsigned int length = std::strlen(text);

    const char *input_node_name = "input";
    const char *output_node_name = "output";

    std::size_t maxIndex = 0;
    if (length > 512) {
        const char *model_path = "../resources/model32.onnx";
        Ort::Session session(env, model_path, session_options);

        int64_t input_node_dims[] = {1, 1, 32, 32};
        size_t input_tensor_size = 1024;
        float input_tensor_values[input_tensor_size];

        for (unsigned int i = 0; i < input_tensor_size; i++) {
            input_tensor_values[i] = static_cast<float>(static_cast<unsigned char>(text[i % length])) / 255.0f;
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                input_tensor_values,input_tensor_size,
                input_node_dims,4);

        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                &input_node_name,&input_tensor,1,
                &output_node_name,1);

        auto output = output_tensors.front().GetTensorMutableData<float>();

        float maxValue = output[0];
        for (std::size_t i = 0; i < 29; ++i) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }
    }
    else {
        const char *model_path = "../resources/model16.onnx";
        Ort::Session session(env, model_path, session_options);

        int64_t input_node_dims[] = {1, 1, 16, 16};
        size_t input_tensor_size = 256;
        float input_tensor_values[input_tensor_size];

        for (unsigned int i = 0; i < input_tensor_size; i++) {
            input_tensor_values[i] = static_cast<float>(static_cast<unsigned char>(text[i % length])) / 255.0f;
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                input_tensor_values,input_tensor_size,
                input_node_dims,4);

        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                &input_node_name,&input_tensor,1,
                &output_node_name,1);

        auto output = output_tensors.front().GetTensorMutableData<float>();

        float maxValue = output[0];
        for (std::size_t i = 0; i < 29; ++i) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }
    }

    switch (maxIndex) {
        case 1:
            return TGLANG_LANGUAGE_C;
        case 2:
            return TGLANG_LANGUAGE_CPLUSPLUS;
        case 3:
            return TGLANG_LANGUAGE_CSHARP;
        case 4:
            return TGLANG_LANGUAGE_CSS;
        case 5:
            return TGLANG_LANGUAGE_DART;
        case 6:
            return TGLANG_LANGUAGE_DOCKER;
        case 7:
            return TGLANG_LANGUAGE_FUNC;
        case 8:
            return TGLANG_LANGUAGE_GO;
        case 9:
            return TGLANG_LANGUAGE_HTML;
        case 10:
            return TGLANG_LANGUAGE_JAVA;
        case 11:
            return TGLANG_LANGUAGE_JAVASCRIPT;
        case 12:
            return TGLANG_LANGUAGE_JSON;
        case 13:
            return TGLANG_LANGUAGE_KOTLIN;
        case 14:
            return TGLANG_LANGUAGE_LUA;
        case 15:
            return TGLANG_LANGUAGE_NGINX;
        case 16:
            return TGLANG_LANGUAGE_OBJECTIVE_C;
        case 17:
            return TGLANG_LANGUAGE_PHP;
        case 18:
            return TGLANG_LANGUAGE_POWERSHELL;
        case 19:
            return TGLANG_LANGUAGE_PYTHON;
        case 20:
            return TGLANG_LANGUAGE_RUBY;
        case 21:
            return TGLANG_LANGUAGE_RUST;
        case 22:
            return TGLANG_LANGUAGE_SHELL;
        case 23:
            return TGLANG_LANGUAGE_SOLIDITY;
        case 24:
            return TGLANG_LANGUAGE_SQL;
        case 25:
            return TGLANG_LANGUAGE_SWIFT;
        case 26:
            return TGLANG_LANGUAGE_TL;
        case 27:
            return TGLANG_LANGUAGE_TYPESCRIPT;
        case 28:
            return TGLANG_LANGUAGE_XML;
        default:
            return TGLANG_LANGUAGE_OTHER;
    }
}