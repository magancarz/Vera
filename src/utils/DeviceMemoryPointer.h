#pragma once

#include <cuda_runtime.h>

namespace dmm
{
    template <typename T>
    class DeviceMemoryPointer
    {
    public:
        DeviceMemoryPointer(size_t num_of_elements)
            : num_of_elements(num_of_elements)
        {
            allocateMemory();
        }

        DeviceMemoryPointer()
            : num_of_elements(1)
        {
            allocateMemory();
        }

        ~DeviceMemoryPointer()
        {
            (*use_count) -= 1;
            freeAllocatedMemoryIfNeeded();
        }

        DeviceMemoryPointer(DeviceMemoryPointer& dmp) noexcept { copy(dmp); }
        DeviceMemoryPointer(const DeviceMemoryPointer& dmp) noexcept { copy(dmp); }
        DeviceMemoryPointer& operator=(const DeviceMemoryPointer& dmp)
    	{
            if (&dmp == this)
            {
                return *this;
            }
            copy(dmp);
            return *this;
        }

        void copyFrom(const T* from)
        {
            cudaMemcpy(ptr, from, num_of_elements * sizeof(T), cudaMemcpyHostToDevice);
        }

        void copyFrom(const T&& from)
        {
            cudaMemcpy(ptr, &from, num_of_elements * sizeof(T), cudaMemcpyHostToDevice);
        }

        void copyFrom(DeviceMemoryPointer& from)
        {
            cudaMemcpy(ptr, from.data(), num_of_elements * sizeof(T), cudaMemcpyHostToDevice);
        }

        __host__ __device__ T* data() { return ptr; }
        __host__ __device__ size_t size() const { return num_of_elements; }

        __host__ __device__ T* operator->() const { return ptr; }
        __host__ __device__ T operator*() { return *ptr; }
        __host__ __device__ T& operator[](size_t index)
        {
            if (index < num_of_elements)
            {
                return ptr[index];
            }
            printf("Tried to get element out of bounds index!\n");
            return ptr[0];
        }

        __host__ __device__ T& operator[](size_t index) const
        {
            if (index < num_of_elements)
            {
                return ptr[index];
            }
            printf("Tried to get element out of bounds index!\n");
            return ptr[0];
        }
    private:
        void allocateMemory()
        {
            cudaMallocManaged(&ptr, num_of_elements * sizeof(T));
            use_count = new unsigned int;
            *use_count = 1;
        }

        void copy(const DeviceMemoryPointer& from)
        {
            ptr = from.ptr;
            num_of_elements = from.num_of_elements;
            use_count = from.use_count;
            (*use_count) += 1;
        }

        void copy(const DeviceMemoryPointer&& from)
        {
            ptr = from.ptr;
            num_of_elements = from.num_of_elements;
            use_count = from.use_count;
            (*use_count) += 1;
        }

        void freeAllocatedMemoryIfNeeded()
        {
            if ((*use_count) <= 0)
            {
                cudaFree(ptr);
                delete use_count;
            }
        }

        T* ptr;
        unsigned int* use_count{nullptr};
        size_t num_of_elements;
    };
}