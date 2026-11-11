// Device compilation belongs to the build, never a measured solve.
#include <nvrtc.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) return 3;
    std::string code((std::istreambuf_iterator<char>(input)), {});
    nvrtcProgram p{};
    if (nvrtcCreateProgram(&p, code.c_str(), argv[1], 0, nullptr, nullptr)) return 4;
    std::string arch = std::string("--gpu-architecture=") + argv[3];
    const char* options[] = {"--std=c++17", arch.c_str(), "--fmad=true"};
    auto result = nvrtcCompileProgram(p, 3, options);
    size_t n = 0; nvrtcGetProgramLogSize(p, &n);
    std::vector<char> log(n); nvrtcGetProgramLog(p, log.data());
    if (n > 1) std::cerr << log.data();
    if (result) { nvrtcDestroyProgram(&p); return 5; }
    bool ptx = std::string(argv[3]).find("compute") == 0;
    result = ptx ? nvrtcGetPTXSize(p, &n) : nvrtcGetCUBINSize(p, &n);
    if (result) { nvrtcDestroyProgram(&p); return 6; }
    std::vector<char> image(n);
    result = ptx ? nvrtcGetPTX(p, image.data()) : nvrtcGetCUBIN(p, image.data());
    nvrtcDestroyProgram(&p);
    if (result) return 7;
    std::ofstream output(argv[2], std::ios::binary);
    output.write(image.data(), image.size());
    if (!output) return 8;
    std::cout << argv[3] << ": " << image.size() << " bytes\n";
    return 0;
}
