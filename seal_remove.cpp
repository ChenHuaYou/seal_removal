#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"

#define USE_INT8  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

// UNet网络的输入输出宽高
static const int UNET_INPUT_H = 512;
static const int UNET_INPUT_W = 512;

// 设置YOLOv5和UNet显存大小
const int YOLOV5_INPUT_SIZE = BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float);
const int YOLOV5_OUTPUT_SIZE = BATCH_SIZE * OUTPUT_SIZE * sizeof(float);
const int UNET_INPUT_SIZE = BATCH_SIZE * 3 * UNET_INPUT_H * UNET_INPUT_W * sizeof(float);
const int UNET_OUTPUT_SIZE = BATCH_SIZE * 3 * UNET_INPUT_H * UNET_INPUT_W * sizeof(float);

static Logger gLogger;


static int get_width(int x, float gw, int divisor = 8) {
    //return math.ceil(x / divisor) * divisor

    return  int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) {
        return 1;
    } else {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}


void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize, int w, int h, int opt_size) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * w * h * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize  * opt_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net[0] == 's') {
            gd = 0.33;
            gw = 0.50;
        } else if (net[0] == 'm') {
            gd = 0.67;
            gw = 0.75;
        } else if (net[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
        } else if (net[0] == 'x') {
            gd = 1.33;
            gw = 1.25;
        } else if (net[0] == 'c' && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
        } else {
            return false;
        }
        if (net.size() == 2 && net[1] == '6') {
            is_p6 = true;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

char* deserializeModel(char * engine_name, size_t& size){
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        abort();
    }
    std::cout << "start deserialize " << engine_name << " !" << std::endl; 
    char *trtModelStream = nullptr;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close(); 
    std::cout << "[ok] finish deserialize " << engine_name << " !" << std::endl;
    return trtModelStream;
}

void iou_cover_img(cv::Mat& input, cv::Rect& r, float* data){
    // int rows = img.rows;
	// int cols = img.cols * img.channels();
    // std::cout << "start write image..." << std::endl;
	cv::Mat dstMat(UNET_INPUT_H, UNET_INPUT_W, CV_8UC3);
	int offset = 0;
	// 遍历data数组将其乘以255.f，然后再转成uchar
	for (int i = 0; i < UNET_INPUT_H; ++i) {
		uchar* pointer = dstMat.ptr<uchar>(i);
		for (int j = 0; j < UNET_INPUT_W * 3; j += 3) {
			// RGB -> BGR
			// 蓝色通道
			pointer[j] = static_cast<uchar>((data[2 * 3 * UNET_INPUT_W + offset] * 0.5 + 0.5) * 255.0);
			pointer[j + 1] = static_cast<uchar>((data[UNET_INPUT_W * UNET_INPUT_W + offset] * 0.5 + 0.5) * 255.0);
			pointer[j + 2] = static_cast<uchar>((data[offset] * 0.5 + 0.5 ) * 255.0);
			++offset;
		}
	}
	if (!dstMat.data) {
		std::cout << "detection is nullptr!" << std::endl;
		abort();
	}
	// cv::imwrite("seal_remove.png", dstMat);
    // 将dstMat进行resize，使其和原来的图像大小相等
    cv::resize(dstMat,dstMat,cv::Size(r.width, r.height));
    // std::cout << "input(r).size() = "<< input(r).size() << std::endl;
    // std::cout << "dstMat.size() = "<< dstMat.size() << std::endl;
    assert(input(r).size() == dstMat.size());
    // input(r) = dstMat;
    dstMat.copyTo(input(r));
    // cv::imwrite("dstMat.png", dstMat);
    // std::cout << "[ok] finish writing image!" << std::endl;
}

void readImage(cv::Mat& img, float* data){
    // cv::Mat img = cv::imread(image);
	// if (!img.data) {
	// 	std::cout << "fail to open img file" << std::endl;
	// 	abort();
	// }
	// std::cout << "cols = " << std::to_string(img.cols) << std::endl;
	// std::cout << "rows = " << std::to_string(img.rows) << std::endl;
    // 对img区域进行resize操作
    cv::resize(img, img, cv::Size(UNET_INPUT_W, UNET_INPUT_H));
    // std::cout << "resized img cols = " << std::to_string(img.cols) << std::endl;
    // std::cout << "resized img rows = " << std::to_string(img.rows) << std::endl;
	int cols = img.cols * img.channels();
	int rows = img.rows;
	if (img.isContinuous()) {
	 	// std::cout << "image is continuous!" << std::endl;
		cols *= rows;
		rows = 1;
	}
	// 将图像的数据保存到一维数组中[红色，绿色，蓝色]
	int offset = 0;
	// static float *data = new float[3 * UNET_INPUT_H * UNET_INPUT_W];
	// 遍历每一行数据
	for (int i = 0; i < rows; ++i) {
		// 得到该行的指针
		uchar* pointer = img.ptr<uchar>(i);
		// 每一列数据（每三步处理一次）
		for (int j = 0; j < cols; j += img.channels()) {
			// 红色数据放到[0, H * W -1]的位置，绿色放到[H*W , 2 * H * W-1]的位置，蓝色放到[2*W*H, 3*W*H-1]的位置
			data[offset] =( static_cast<float>(pointer[j + 2]) / 255.0 - 0.5) / 0.5;
			data[offset + UNET_INPUT_H * UNET_INPUT_W] = (static_cast<float>(pointer[j + 1]) / 255.0 - 0.5) / 0.5;
			data[offset + 2 * UNET_INPUT_H * UNET_INPUT_W] = (static_cast<float>(pointer[j]) / 255.0 - 0.5) / 0.5;
			++offset;
		}
	}
    assert(data != nullptr);
    // std::cout << "[ok] finish read image!" << std::endl;
}

int main(int argc, char** argv) {

    // ubuntu终端中运行
    // fp32: ./seal_remove  ../yolov5m_fp32.engine ../unet_fp32.engine ../data
    // fp16: ./seal_remove  ../yolov5m_fp16.engine ../unet_fp16.engine ../data
    // int8: ./seal_remove  ../yolov5m_int8.engine ../unet_int8.engine ../data

    cudaSetDevice(DEVICE);

    // 对yolov4模型和unet模型进行反序列化
    std::string img_dir = argv[3];
    size_t yolov5_size = 0, unet_size = 0;
    char* yolov5_stream = deserializeModel(argv[1], yolov5_size);
    char* unet_stream = deserializeModel(argv[2], unet_size);
    assert(yolov5_size > 0);
    assert(unet_size > 0);

    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    static float* yolov5_data = new float[YOLOV5_INPUT_SIZE];
    static float* unet_data = new float[UNET_INPUT_SIZE];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float* yolov5_prob = new float[YOLOV5_OUTPUT_SIZE];
    static float* unet_prob = new float[UNET_OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* yolov5_engine = runtime->deserializeCudaEngine(yolov5_stream, yolov5_size);
    ICudaEngine* unet_engine = runtime->deserializeCudaEngine(unet_stream, unet_size);
    assert(yolov5_engine != nullptr);
    assert(unet_engine != nullptr);
    IExecutionContext* yolov5_context = yolov5_engine->createExecutionContext();
    IExecutionContext* unet_context = unet_engine->createExecutionContext();
    assert(yolov5_context != nullptr);
    assert(unet_context != nullptr);
    delete[] yolov5_stream;
    delete[] unet_stream;
    assert(yolov5_engine->getNbBindings() == 2);
    assert(unet_engine->getNbBindings() == 2);
    void* yolov5_buffers[2], *unet_buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int yolov5_inputIndex = yolov5_engine->getBindingIndex(INPUT_BLOB_NAME);
    const int yolov5_outputIndex = yolov5_engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(yolov5_inputIndex == 0);
    assert(yolov5_outputIndex == 1);
    // 确保unet的输入输出是符合要求的
    const int unet_inputIndex = unet_engine->getBindingIndex(INPUT_BLOB_NAME);
    const int unet_outputIndex = unet_engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(unet_inputIndex == 0);
    assert(unet_outputIndex == 1);
    
    // Create GPU buffers on device
    std::cout << "start malloc for yolov5!" << std::endl; 
    CUDA_CHECK(cudaMalloc(&yolov5_buffers[yolov5_inputIndex], YOLOV5_INPUT_SIZE));
    CUDA_CHECK(cudaMalloc(&yolov5_buffers[yolov5_outputIndex], YOLOV5_OUTPUT_SIZE));
    std::cout << "[ok] finish mallocing for yolov5!" << std::endl; 
    // 为unet创建输入输出显存
    std::cout << "start malloc for unet!" << std::endl; 
    CUDA_CHECK(cudaMalloc(&unet_buffers[unet_inputIndex], UNET_INPUT_SIZE));
    CUDA_CHECK(cudaMalloc(&unet_buffers[unet_outputIndex], UNET_OUTPUT_SIZE));
    std::cout << "[ok] finish mallocing for unet!" << std::endl; 
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int fcount = 0;
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    yolov5_data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                    yolov5_data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                    yolov5_data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }

        // Run inference
        std::cout << "yolov5 start inference!" << std::endl; 
        auto start = std::chrono::system_clock::now();
        doInference(*yolov5_context, stream, yolov5_buffers, yolov5_data, yolov5_prob, BATCH_SIZE, INPUT_W, INPUT_H, OUTPUT_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << "yolov5 consuming time: " <<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::cout << "[ok] yolov5 has finished inferencing!" << std::endl; 
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &yolov5_prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            //std::cout << res.size() << std::endl;
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            // 一共有res.size()个矩形框
            std::cout << "unet start inference!" << std::endl;
            start = std::chrono::system_clock::now();
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(img, res[j].bbox);
                // 通过矩形框得到感兴趣的区域
                cv::Mat roi;
                img(r).copyTo(roi);
                // 使用gpu对unet进行推理加速
                readImage(roi, unet_data);
                doInference(*unet_context, stream, unet_buffers, unet_data, unet_prob, BATCH_SIZE, UNET_INPUT_W, UNET_INPUT_H, 3 * UNET_INPUT_W * UNET_INPUT_H);
                // TODO: 将unet_prob转成mat形式，然后再覆盖原图中rect的位置
                iou_cover_img(img, r, unet_prob);
            }
            end = std::chrono::system_clock::now();
            std::cout << "unet consuming time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
            std::cout << "[ok] unet has finished inferencing!" << std::endl;
            cv::imwrite("../_repaired_int8_" + file_names[f - fcount + 1 + b], img);
        }
        fcount = 0;
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(yolov5_buffers[yolov5_inputIndex]));
    CUDA_CHECK(cudaFree(yolov5_buffers[yolov5_outputIndex]));
    // 对Unet网络用的显存进行释放
    CUDA_CHECK(cudaFree(unet_buffers[unet_inputIndex]));
    CUDA_CHECK(cudaFree(unet_buffers[unet_outputIndex]));
    // Destroy the engine
    yolov5_context->destroy();
    unet_context->destroy();
    yolov5_engine->destroy();
    unet_engine->destroy();
    runtime->destroy();


    return 0;
}