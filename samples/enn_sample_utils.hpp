#include <iostream>
#include <cmath>

#define RED "\x1b[31m"
#define GREEN "\x1b[32m"
#define YELLOW "\x1b[33m"
#define CYAN "\x1b[36m"
#define RESET "\x1b[0m"

#define PRINT_AND_RETURN(message, ...)       \
    printf("TEST: " message, ##__VA_ARGS__); \
    printf("\n");                            \
    return -1;

#define PRINT(message, ...)                  \
    printf("TEST: " message, ##__VA_ARGS__); \
    printf("\n");

#define TEST_FILE_PATH "/data/vendor/enn/models/"
#define TEST_FILE(M) ((TEST_FILE_PATH + std::string(M)).c_str())

namespace enn {
namespace sample_utils {

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline static void show_diff(const char *title, int idx, T s, T t, T diff, T threshold) {
    std::cout << "[" << idx << "] " << title << ": golden " << s << " vs. target " << t << " = " << diff << " > threshold "
              << threshold << std::endl;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
inline static void show_diff(const char *title, int idx, T s, T t, T diff, T threshold) {
    std::cout << "[" << idx << "] " << title << ": golden " << s << " vs. target " << t << " = " << diff << " > threshold "
              << threshold << std::endl;
}

inline static void show_diff(const char *title, int idx, uint8_t s, uint8_t t, uint8_t diff, uint8_t threshold) {
    std::cout << "[" << idx << "] " << title << ": golden " << s << " vs. target " << t << " = " << diff << " > threshold "
              << threshold << std::endl;
}

template <typename CompareUnitType>
int32_t CompareBuffersWithThreshold(void *goldenBuffer, void *TargetBufferAddr, int32_t buffer_size,
                                    uint8_t *compare_buffer_out = nullptr, CompareUnitType threshold = 0,
                                    bool is_debug = false) {
    CompareUnitType *target_p = reinterpret_cast<CompareUnitType *>(TargetBufferAddr);
    CompareUnitType *source_p = reinterpret_cast<CompareUnitType *>(goldenBuffer);
    bool is_save = (compare_buffer_out != nullptr);
    int diff_cnt = 0;

    if (is_save) {
        for (size_t i = 0; i < buffer_size / sizeof(CompareUnitType); i++) {
            compare_buffer_out[i] = 0;
        }
    }

    for (size_t i = 0; i < buffer_size / sizeof(CompareUnitType); i++) {
        auto diff = std::abs(source_p[i] - target_p[i]);
        if (diff > threshold) {
            diff_cnt++;
            if (is_save) {
                compare_buffer_out[i] = 0xFF;
            } else {
                show_diff("Different!", i, source_p[i], target_p[i], diff, threshold);
            }
        }
        if (is_debug) {
            show_diff("Check", i, source_p[i], target_p[i], diff, threshold);
        }
    }

    return diff_cnt;
}

int get_file_size(const char *filename) {
    FILE *f = fopen(filename, "rb");

    if (f == nullptr) {
        std::cerr << "File open Error" << std::endl;
        return -1;
    }

    fseek(f, 0, SEEK_END);
    int size = ftell(f);
    fseek(f, 0, SEEK_SET);
    fclose(f);

    return size;
}

int import_file_to_mem(const char *filename, char *target_va) {
    auto file_size = get_file_size(filename);
    if (file_size < 0) {
        std::cerr << "Wrong file size!: " << file_size << std::endl;
        return -1;
    }

    FILE *f = fopen(filename, "rb");

    if (file_size != static_cast<int>(fread(target_va, sizeof(char), file_size, f))) {
        std::cerr << "File fread Error!: " << filename << ", size: " << file_size << std::endl;
        return -1;
    }

    fclose(f);

    return 0;
}

int export_mem_to_file(const char *filename, const void *va, uint32_t size) {
    size_t ret_cnt;

    PRINT("DEBUG:: Export memory to file: name(%s) va(%p), size(%d)", filename, va, size);

    FILE *fp = fopen(filename, "wb");

    ret_cnt = fwrite(va, size, 1, fp);
    if (ret_cnt <= 0) {
        PRINT("FileWrite Failed!!(%zu)", ret_cnt);
        fclose(fp);
        return ENN_RET_INVAL;
    }

    PRINT("DEBUG:: File Save Completed.");
    fclose(fp);

    return ENN_RET_SUCCESS;
}

void show_raw_memory_to_hex(uint8_t *va, uint32_t size, const int line_max, const uint32_t size_max) {
    char line_tmp[100] = {0,};
    int int_size = static_cast<int>(size);
    int max = (size_max == 0 ? int_size : (int_size < size_max ? int_size : size_max));
    int idx = sprintf(line_tmp, "[%p] ", va);  // prefix of line
    int i = 0;                                 // idx records current location of print line
    for (; i < max; ++i) {
        idx += sprintf(&(line_tmp[idx]), "%02X ", va[i]);
        if (i % line_max == (line_max - 1)) {
            // if new line is required, flush print --> and record prefix print
            line_tmp[idx] = 0;
            std::cout << line_tmp << std::endl;
            idx = 0;
            idx = sprintf(line_tmp, "[%p] ", &(va[i]));
        }
    }
    if (i % line_max != 0) {
        std::cout << line_tmp << std::endl;
    }
}

int check_valid_buffers_info(NumberOfBuffersInfo &buffers_info, int input_num, int output_num) {
    uint32_t n_in_buf = buffers_info.n_in_buf;
    uint32_t n_out_buf = buffers_info.n_out_buf;

    if (n_in_buf != input_num) {
        PRINT_AND_RETURN("Invalid input number Error");
    }

    if (n_out_buf != output_num) {
        PRINT_AND_RETURN("Invalid output number Error");
    }

    PRINT("# Input: %d, Output: %d", n_in_buf, n_out_buf);

    return 0;
}

int load_input_files(int idx_start, int idx_end, const std::vector<std::string> &files, EnnBufferPtr *buffers) {
    for (int i = idx_start; i < idx_end; ++i) {
        PRINT("[%2d] IFM: va: %p, size: %7d", i, buffers[i]->va, buffers[i]->size);

        if (enn::sample_utils::import_file_to_mem(files[i].c_str(), reinterpret_cast<char *>(buffers[i]->va))) {
            PRINT_AND_RETURN("Failed to load Input from file");
        }
    }

    return 0;
}

int load_golden_files_and_compare(int idx_start, int idx_end, const std::vector<std::string> &files, EnnBufferPtr *buffers,
                                  float threshold = 0.1f) {
    for (int i = idx_start; i < idx_end; ++i) {
        PRINT("[%2d] OFM: va: %p, size: %7d", i - idx_start, buffers[i]->va, buffers[i]->size);

        char *golden_out_ref = new char[buffers[i]->size];
        if (enn::sample_utils::import_file_to_mem(files[i - idx_start].c_str(), golden_out_ref)) {
            delete[] golden_out_ref;
            PRINT_AND_RETURN("Failed to load Golden Output from file");
        }

        // compare output to golden out
        auto diff_pixels = enn::sample_utils::CompareBuffersWithThreshold<float>(
            reinterpret_cast<void *>(golden_out_ref), buffers[i]->va, buffers[i]->size, nullptr, threshold);

        if (diff_pixels == 0) {
            PRINT(GREEN "     [SUCCESS] Result is matched with golden out" RESET);
        } else {
            PRINT(RED "     [FAILED ] Result is not matched with golden out" RESET);
        }

        delete[] golden_out_ref;
    }

    return 0;
}

}  // namespace sample_utils
}  // namespace enn
