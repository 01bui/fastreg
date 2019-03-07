#pragma once

/* fastreg includes */
#include <safe_call.hpp>
#include <types.hpp>

/* sys headers */
#include <string>

namespace fastreg {
namespace cuda {

int getCudaEnabledDeviceCount();
void setDevice(int device);
std::string getDeviceName(int device);

void printCudaDeviceInfo(int device);
void printShortCudaDeviceInfo(int device);

}  // namespace cuda
}  // namespace fastreg
