//We'll call this the Stendhal Red Black Scheme.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "C:\Users\Philadelphia\Documents\1_SweptTimeResearch\CUDA Examples\cuda_by_example\common\book.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <cmath>
#include <ctime>
#include <stdlib.h>
#include <fstream>
#include <iostream>

using namespace std;

// Now the plate must be square.  #Divisions should be some multiple of 32, preferably some 2^x.
#define LENS       5.
#define TH_COND   16.
#define DZ        .01
#define DIVISIONS 2048
#define TOLERANCE 1e-5

// 
__constant__ double d_bc[4];
__constant__ double d_guess;
__constant__ double d_A;
__constant__ double d_ds;
__constant__ double d_d_coeff;

__device__ void *cornerSource (int BC1, int BC2, double *source)
{
    if (BC1>0)
    {
        if (BC2>0)
        {
            source[0] = 2*d_d_coeff*(BC1 + BC2);
            source[1] = -4*d_coeff;
        }
        else
        {
            source[0] = 2*d_coeff*BC1;
            source[1] = -2*d_coeff;
        }
    }
    else if (BC2>0)
    {
        source[0] = 2*d_coeff*BC2;
        source[1] = -2*d_coeff;
    }
    else
    {
        source[0] = 0;
        source[1] = 0;
    }
}

__device__ void differencingOperation(double *active_half[][DIVISIONS/2], double *passive_half[][DIVISIONS/2])
{
	
    double d_coeff_p;
    double *source = new double[2];
	for (int i = 0; i < rw; i++)
            {
                for (int j = (m+i)%2; j < cl; j+=2)
                {
                    if (i == 0)
                    {
                        if (j == 0)
                        {
                            cornerSource(BC[2],BC[3],d_coeff, source);
                            d_coeff_p = 2*d_coeff-source[1];
                            t_matrix_final[i][j] = (d_coeff*(t_matrix_final[i+1][j]+t_matrix_final[i][j+1])+source[0])/d_coeff_p;
                        }
                        else if (j == (cl-1))
                        {
                            cornerSource(BC[2],BC[1],d_coeff, source);
                            d_coeff_p = 2*d_coeff-source[1];
                            t_matrix_final[i][j] = (d_coeff*(t_matrix_final[i+1][j]+t_matrix_final[i][j-1])+source[0])/d_coeff_p;
                        }
                        else
                        {
                            if (BC[2]>0)
                            {
                                source[0] = 2*d_coeff*BC[2];
                                source[1] = -2*d_coeff;
                                d_coeff_p = 3*d_coeff-source[1];
                                t_matrix_final[i][j] = (d_coeff*(t_matrix_final[i][j+1]+t_matrix_final[i+1][j]+t_matrix_final[i][j-1])+source[0])/d_coeff_p;
                            }
                            else
                            {
                                d_coeff_p = 3*d_coeff;
                                t_matrix_final[i][j] = d_coeff*(t_matrix_final[i+1][j]+t_matrix_final[i][j-1]+t_matrix_final[i][j+1])/d_coeff_p;

                            }
                        }
                    }
                    else if (j == 0)
                    {
                        if (i == (rw-1))
                        {
                            cornerSource(BC[0],BC[3],d_coeff, source);
                            d_coeff_p = 2*d_coeff-source[1];
                            t_matrix_final[i][j] = (d_coeff*(t_matrix_final[i-1][j] + t_matrix_final[i][j+1])+source[0])/d_coeff_p;
                        }
                        else
                        {
                            if (BC[3]>0)
                            {
                                source[0] = 2*d_coeff*BC[3];
                                source[1] = -2*d_coeff;
                                d_coeff_p = 3*d_coeff-source[1];
                                t_matrix_final[i][j] = (d_coeff*(t_matrix_final[i-1][j] + t_matrix_final[i+1][j] + t_matrix_final[i][j+1])+source[0])/d_coeff_p;
                            }
                            else
                            {
                                d_coeff_p = 3*d_coeff;
                                t_matrix_final[i][j] = d_coeff*(t_matrix_final[i+1][j]+t_matrix_final[i-1][j]+t_matrix_final[i][j+1])/d_coeff_p;

                            }
                        }
                    }
                    else if (i == (rw-1))
                    {
                        if (j == (cl-1))
                        {
                            cornerSource(BC[0], BC[1], d_coeff, source);
                            d_coeff_p = 2*d_coeff-source[1];
                            t_matrix_final[i][j] = (d_coeff*(t_matrix_final[i-1][j]+t_matrix_final[i][j-1])+source[0])/d_coeff_p;

                        }
                        else
                        {
                            if (BC[0]>0)
                            {
                                source[0] = 2*d_coeff*BC[0];
                                source[1] = -2*d_coeff;
                                d_coeff_p = 3*d_coeff-source[1];
                                t_matrix_final[i][j] = (d_coeff*(t_matrix_final[i-1][j]+t_matrix_final[i][j-1]+t_matrix_final[i][j+1])+source[0])/d_coeff_p;
                            }
                            else
                            {
                                d_coeff_p = 3*d_coeff;
                                t_matrix_final[i][j] = d_coeff*(t_matrix_final[i-1][j]+t_matrix_final[i][j-1]+t_matrix_final[i][j+1])/d_coeff_p;

                            }
                        }
                    }
                    else if (j == (cl-1))
                    {
                        if (BC[1]>0)
                        {
                            source[0] = 2*d_coeff*BC[1];
                            source[1] = -2*d_coeff;
                            d_coeff_p = 3*d_coeff-source[1];
                            t_matrix_final[i][j] = (d_coeff*(t_matrix_final[i-1][j]+t_matrix_final[i+1][j]+t_matrix_final[i][j-1])+source[0])/d_coeff_p;
                        }
                        else
                        {
                            d_coeff_p = 3*d_coeff;
                            t_matrix_final[i][j] = d_coeff*(t_matrix_final[i-1][j]+t_matrix_final[i+1][j]+t_matrix_final[i][j-1])/d_coeff_p;

                        }
                    }
                    else
                    {
                        d_coeff_p = 4*d_coeff;
                        t_matrix_final[i][j] = d_coeff*(t_matrix_final[i-1][j]+t_matrix_final[i+1][j]+t_matrix_final[i][j-1]+t_matrix_final[i][j+1])/d_coeff_p;
                    }
                }
            }
        }


__global__ void gaussSeidel_RB ( )
{
    // Initialize array of guesses.
    double red_matrix_initial[DIVISIONS][DIVISIONS/2];
    double red_matrix_final[DIVISIONS][DIVISIONS/2];
    double black_matrix_initial[DIVISIONS][DIVISIONS/2];
    double black_matrix_final[DIVISIONS][DIVISIONS/2];

    double diff_matrix;
    int iter = 0;

    // Initialize matrices.
	red_matrix_initial[blockIdx.x][threadIdx.x] = d_guess;
	__syncthreads();
	red_matrix_final[blockIdx.x][threadIdx.x] = d_guess;
	__syncthreads();
	black_matrix_initial[blockIdx.x][threadIdx.x] = d_guess;
	__syncthreads();
	black_matrix_final[blockIdx.x][threadIdx.x] = d_guess;
	__syncthreads();


    while (true)
    {

        differencingOperation(red_matrix_final, black_matrix_final);
		__syncthreads();
		differencingOperation(black_matrix_final, red_matrix_final);
		__syncthreads();

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
        //    ofstream filewrite;
        //    filewrite.open("C:\\Users\\Philadelphia\\Documents\\1_SweptTimeResearch\\GaussSeidel\\GaussSeidelCPP\\GS_output.txt", ios::trunc);
        //    filewrite << LENX << "\n" << LENY << "\n" << DS;

        //    for (int k = 0; k < rw; k++)
        //    {
        //        for (int n = 0; n < cl; n++)
        //        {
        //            filewrite << "\n" << t_matrix_final[k][n];
        //        }
        //    }

        //    filewrite.close();
        //    free(source);
        //    
        //}

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

    double ds = LENS/((double)(DIVISIONS-1));
    double A = DZ * ds;
    double guess;
	double coeff = TH_COND * d_A / d_ds;
    double bound_cond[4];


    // Get initial conditions
    cout << "Provide Boundary conditions for each edge of the slab.\nEnter Constant Temperature in KELVIN\nor a negative number for an insulated boundary:\nNorth: \n";
	cin >> bound_cond[0];
    cout << "East: \n";
    cin >> bound_cond[1];
    cout << "South: \n";
    cin >> bound_cond[2];
    cout << "West: \n";
    cin >> bound_cond[3];

    // Get Guess for slab temperature
    cout << "Provide a guess Temperature for the slab in Kelvin:\n";
    cin >> guess;

	// Initialize the CUDA put the constants in constant memory
	cudaMemcpyToSymbol( d_bc, bound_cond, sizeof(bound_cond));
	cudaMemcpyToSymbol( &d_guess, &guess, sizeof(guess)) ;
	cudaMemcpyToSymbol( &d_A, &A, sizeof(A));
	cudaMemcpyToSymbol( &d_ds, &ds, sizeof(ds));
	cudaMemcpyToSymbol( &d_d_coeff, &coeff, sizeof(coeff));

    double wall0 = clock();
    gaussSeidel_RB <<< DIVISIONS,DIVISIONS/2 >>> ();
    double wall1 = clock();
    double timed = (wall1-wall0)/double(CLOCKS_PER_SEC);

    printf("That took %.3e seconds.\n",timed);

	double red;
	double black;


	cudaFree(&d_A);
	cudaFree(&d_ds);
	cudaFree(&d_guess);
	cudaFree(&d_bc);

    return 0;
}
