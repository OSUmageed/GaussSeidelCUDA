
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <stdlib.h>

#define LENX      5.
#define LENY      2.
#define TH_COND   16.
#define DZ        .01
#define DS        .05
#define TOLERANCE 1e-5

struct BoundaryTemperature
{
    int North;
    int East;
    int South;
    int West;

};

// Instead of writing this over and over again.  Is this a leak?
__device__ void *cornerSource (int BC1, int BC2, double coeff, double *source)
{
    if (BC1>0)
    {
        if (BC2>0)
        {
            source[0] = 2*coeff*(BC1 + BC2);
            source[1] = -4*coeff;
        }
        else
        {
            source[0] = 2*coeff*BC1;
            source[1] = -2*coeff;
        }
    }
    else if (BC2>0)
    {
        source[0] = 2*coeff*BC2;
        source[1] = -2*coeff;
    }
    else
    {
        source[0] = 0;
        source[1] = 0;
    }
}

// I think we're going to return the number of iterations it took rather than the result which I guess would be read out.
__global__ int gaussSeidel_RB (double A, BoundaryTemperature BC, double guess, int rw, int cl)
{
    // Initialize array of guesses.
    double t_matrix_initial[rw][cl];
    double t_matrix_final[rw][cl];
    double coeff = TH_COND * A / DS;
    double coeff_p;
    double *source = new double[2];
    double diff_matrix;
    int iter = 0;

    // For loop to initialize matrices.
    for (int k = 0; k < rw; k++)
    {
        for (int n = 0; n < cl; n++)
        {
            t_matrix_initial[k][n] = guess;
            t_matrix_final[k][n] = guess;
        }
    }

    while (true)
    {
        for (int m = 0; m < 2; m++)
        {
            for (int i = 0; i < rw; i++)
            {
                for (int j = (m+i)%2; j < cl; j+=2)
                {
                    if (i == 0)
                    {
                        if (j == 0)
                        {
                            cornerSource(BC.South,BC.West,coeff, source);
                            coeff_p = 2*coeff-source[1];
                            t_matrix_final[i][j] = (coeff*(t_matrix_final[i+1][j]+t_matrix_final[i][j+1])+source[0])/coeff_p;
                        }
                        else if (j == (cl-1))
                        {
                            cornerSource(BC.South,BC.East,coeff, source);
                            coeff_p = 2*coeff-source[1];
                            t_matrix_final[i][j] = (coeff*(t_matrix_final[i+1][j]+t_matrix_final[i][j-1])+source[0])/coeff_p;
                        }
                        else
                        {
                            if (BC.South>0)
                            {
                                source[0] = 2*coeff*BC.South;
                                source[1] = -2*coeff;
                                coeff_p = 3*coeff-source[1];
                                t_matrix_final[i][j] = (coeff*(t_matrix_final[i][j+1]+t_matrix_final[i+1][j]+t_matrix_final[i][j-1])+source[0])/coeff_p;
                            }
                            else
                            {
                                coeff_p = 3*coeff;
                                t_matrix_final[i][j] = coeff*(t_matrix_final[i+1][j]+t_matrix_final[i][j-1]+t_matrix_final[i][j+1])/coeff_p;

                            }
                        }
                    }
                    else if (j == 0)
                    {
                        if (i == (rw-1))
                        {
                            cornerSource(BC.North,BC.West,coeff, source);
                            coeff_p = 2*coeff-source[1];
                            t_matrix_final[i][j] = (coeff*(t_matrix_final[i-1][j] + t_matrix_final[i][j+1])+source[0])/coeff_p;
                        }
                        else
                        {
                            if (BC.West>0)
                            {
                                source[0] = 2*coeff*BC.West;
                                source[1] = -2*coeff;
                                coeff_p = 3*coeff-source[1];
                                t_matrix_final[i][j] = (coeff*(t_matrix_final[i-1][j] + t_matrix_final[i+1][j] + t_matrix_final[i][j+1])+source[0])/coeff_p;
                            }
                            else
                            {
                                coeff_p = 3*coeff;
                                t_matrix_final[i][j] = coeff*(t_matrix_final[i+1][j]+t_matrix_final[i-1][j]+t_matrix_final[i][j+1])/coeff_p;

                            }
                        }
                    }
                    else if (i == (rw-1))
                    {
                        if (j == (cl-1))
                        {
                            cornerSource(BC.North, BC.East, coeff, source);
                            coeff_p = 2*coeff-source[1];
                            t_matrix_final[i][j] = (coeff*(t_matrix_final[i-1][j]+t_matrix_final[i][j-1])+source[0])/coeff_p;

                        }
                        else
                        {
                            if (BC.North>0)
                            {
                                source[0] = 2*coeff*BC.North;
                                source[1] = -2*coeff;
                                coeff_p = 3*coeff-source[1];
                                t_matrix_final[i][j] = (coeff*(t_matrix_final[i-1][j]+t_matrix_final[i][j-1]+t_matrix_final[i][j+1])+source[0])/coeff_p;
                            }
                            else
                            {
                                coeff_p = 3*coeff;
                                t_matrix_final[i][j] = coeff*(t_matrix_final[i-1][j]+t_matrix_final[i][j-1]+t_matrix_final[i][j+1])/coeff_p;

                            }
                        }
                    }
                    else if (j == (cl-1))
                    {
                        if (BC.East>0)
                        {
                            source[0] = 2*coeff*BC.East;
                            source[1] = -2*coeff;
                            coeff_p = 3*coeff-source[1];
                            t_matrix_final[i][j] = (coeff*(t_matrix_final[i-1][j]+t_matrix_final[i+1][j]+t_matrix_final[i][j-1])+source[0])/coeff_p;
                        }
                        else
                        {
                            coeff_p = 3*coeff;
                            t_matrix_final[i][j] = coeff*(t_matrix_final[i-1][j]+t_matrix_final[i+1][j]+t_matrix_final[i][j-1])/coeff_p;

                        }
                    }
                    else
                    {
                        coeff_p = 4*coeff;
                        t_matrix_final[i][j] = coeff*(t_matrix_final[i-1][j]+t_matrix_final[i+1][j]+t_matrix_final[i][j-1]+t_matrix_final[i][j+1])/coeff_p;
                    }
                }
            }
        }


        // Get the absolute value of the difference.
        diff_matrix = 0;
        for (int k = 0; k < rw; k++)
        {
            for (int n = 0; n < cl; n++)
            {
                diff_matrix += abs(t_matrix_initial[k][n] - t_matrix_final[k][n]);

            }
        }

        iter++;

        if ((diff_matrix / (rw*cl)) < TOLERANCE)
        {
            printf("The solution converges after %.d iterations.\n",iter);

            // Write out matrix to text to read into a sensible language for plotting.
            // Write out the slab dimensions and the Temperatures in 1-D, it'll make it easy to reshape the matrix.
            ofstream filewrite;
            filewrite.open("C:\\Users\\Philadelphia\\Documents\\1_SweptTimeResearch\\GaussSeidel\\GaussSeidelCPP\\GS_output.txt", ios::trunc);
            filewrite << LENX << "\n" << LENY << "\n" << DS;

            for (int k = 0; k < rw; k++)
            {
                for (int n = 0; n < cl; n++)
                {
                    filewrite << "\n" << t_matrix_final[k][n];
                }
            }

            filewrite.close();
            free(source);
            return 1;
        }

        // Make the initial the final.
        for (int k = 0; k < rw; k++)
        {
            for (int n = 0; n < cl; n++)
            {
                t_matrix_initial[k][n] = t_matrix_final[k][n];
            }
        }
    }
}

int main()
{
    int rw = int(LENX/DS + 1);
    int cl = int(LENY/DS + 1);
    double A = DZ * DS;
    double guess;
    BoundaryTemperature bound_cond;

    // Get initial conditions
    cout << "Provide Boundary conditions for each edge of the slab.\nEnter Constant Temperature in KELVIN\nor a negative number for an insulated boundary:\nNorth: \n";
    cin >> bound_cond.North;
    cout << "East: \n";
    cin >> bound_cond.East;
    cout << "South: \n";
    cin >> bound_cond.South;
    cout << "West: \n";
    cin >> bound_cond.West;

    // Get Guess for slab temperature
    cout << "Provide a guess Temperature for the slab in Kelvin:\n";
    cin >> guess;

	// Initialize the CUDA part
	dim3 grid(rw,cl);


    double wall0 = clock();
    gaussSeidel_RB(A, bound_cond, guess, rw, cl);
    double wall1 = clock();
    double timed = (wall1-wall0)/CLOCKS_PER_SEC;

    cout << "That took " << timed << " seconds." << endl;

    return 0;
}

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
