//We'll call this the Stendhal Red Black Scheme.

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda_runtime_api.h"
#include "sm_20_atomic_functions.h"
#include "sm_32_atomic_functions.h"
#include "sm_35_atomic_functions.h"
#include "device_atomic_functions.h"
#include "device_atomic_functions.hpp"
#include "sm_32_atomic_functions.hpp"
#include "sm_20_atomic_functions.hpp"

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
#define TOLERANCE 1.e-5

// Define constants for GPU constant memory.
__device__ __constant__ double d_bc[4];
//__device__ __constant__ double d_guess;
//__device__ __constant__ double d_A;
//__device__ __constant__ double d_ds;
//__device__ __constant__ double d_coeff;

const int x_dim = DIVISIONS/2;
double red[x_dim][DIVISIONS];
double black[x_dim][DIVISIONS];

__device__ void cornerSource (int BC1, int BC2, double *source, double d_coeff)
{
    if (BC1>0)
    {
        if (BC2>0)
        {
            source[0] = 2*d_coeff*(BC1 + BC2);
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

__device__ void differencingOperation(double active_half[][DIVISIONS], double passive_half[][DIVISIONS], int ind_x, int ind_y, int turn, double d_coeff)
{
	
    double d_coeff_p;
    double *source = new double[2];
	// Negative seq means active half starts second.  Positive seq means active half starts first.
	int seq = powf(-1,turn+ind_y);

	// This is to catch indices outside the bounds.

	// If bottom row.
	if (ind_y == 0)
	{
		// If bottom left (SouthWest) corner and red.
		if (ind_x == 0 && turn == 1)
		{
			printf("This Happened!");
			cornerSource(d_bc[2],d_bc[3], source, d_coeff);
			d_coeff_p = 2*d_coeff-source[1];	
			active_half[ind_x][ind_y] = (d_coeff*(passive_half[ind_x][ind_y] + passive_half[ind_x][ind_y+1])+source[0])/d_coeff_p;
		}
		// If bottom right (SouthEast) corner and black.
		else if (ind_x == (DIVISIONS/2-1) && turn == 2)
		{
			cornerSource(d_bc[2],d_bc[1], source, d_coeff);
			d_coeff_p = 2*d_coeff-source[1];
			active_half[ind_x][ind_y] = (d_coeff*(passive_half[ind_x][ind_y] + passive_half[ind_x][ind_y+1])+source[0])/d_coeff_p;
		}
		// Bottom row no corner.
		else
		{
			// Check South Boundary Condition.  If it's constant temperature:
			if (d_bc[2]>0)
			{
				source[0] = 2*d_coeff*d_bc[2];
				source[1] = -2*d_coeff;
				d_coeff_p = 3*d_coeff-source[1];
				active_half[ind_x][ind_y] = (d_coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y+1]+passive_half[ind_x+seq][ind_y])+source[0])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_coeff;
				active_half[ind_x][ind_y] = d_coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y+1]+passive_half[ind_x+seq][ind_y])/d_coeff_p;

			}
		}
	}
	// If top row
	else if (ind_y == DIVISIONS-1)
	{
		// If top right (NorthEast) corner and red.
		if (ind_x == (DIVISIONS/2-1) && turn == 1)
		{
			cornerSource(d_bc[0],d_bc[1], source, d_coeff);
			d_coeff_p = 2*d_coeff-source[1];
			active_half[ind_x][ind_y] = (d_coeff*(passive_half[ind_x][ind_y] + passive_half[ind_x][ind_y-1])+source[0])/d_coeff_p;	
		}
		// If top left (NorthWest) corner and black.
		else if (ind_x == 0 && turn == 2)
		{
			cornerSource(d_bc[0],d_bc[3], source, d_coeff);
			d_coeff_p = 2*d_coeff-source[1];
			active_half[ind_x][ind_y] = (d_coeff*(passive_half[ind_x][ind_y] + passive_half[ind_x][ind_y-1])+source[0])/d_coeff_p;
		}
		// Top row no corner.  The top row is the compliment of the bottom row so the operation for seq is reversed.
		else
		{
			// Check North Boundary Condition.  If it's constant temperature:
			if (d_bc[0]>0)
			{
				source[0] = 2*d_coeff*d_bc[0];
				source[1] = -2*d_coeff;
				d_coeff_p = 3*d_coeff-source[1];
				active_half[ind_x][ind_y] = (d_coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y-1]+passive_half[ind_x-seq][ind_y])+source[0])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_coeff;
				active_half[ind_x][ind_y] = d_coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y-1]+passive_half[ind_x-seq][ind_y])/d_coeff_p;
			}
		}
	}
	// Check side walls.  This is West when the matrix starts the row, that's when seq is -1.
	else if (ind_x == 0 && seq == -1)
	{
		if (d_bc[3]>0)
		{
			source[0] = 2*d_coeff*d_bc[3];
			source[1] = -2*d_coeff;
			d_coeff_p = 3*d_coeff-source[1];
			active_half[ind_x][ind_y] = (d_coeff*(passive_half[ind_x][ind_y]+ passive_half[ind_x][ind_y+1] + passive_half[ind_x][ind_y-1])+source[0])/d_coeff_p;
		}
		else
		{
			d_coeff_p = 3*d_coeff;
			active_half[ind_x][ind_y] = d_coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y+1]+passive_half[ind_x][ind_y-1])/d_coeff_p;

		}
	}
	
	// This is East when the matrix ends the row.
	else if (ind_x == (DIVISIONS/2-1) && seq == 1)
	{
		if (d_bc[1]>0)
		{
			source[0] = 2*d_coeff*d_bc[1];
			source[1] = -2*d_coeff;
			d_coeff_p = 3*d_coeff-source[1];
			active_half[ind_x][ind_y] = (d_coeff*(passive_half[ind_x][ind_y]+ passive_half[ind_x][ind_y+1] + passive_half[ind_x][ind_y-1])+source[0])/d_coeff_p;
		}
		else
		{
			d_coeff_p = 3*d_coeff;
			active_half[ind_x][ind_y] = d_coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y+1]+passive_half[ind_x][ind_y-1])/d_coeff_p;
		}
	}
	// Every cell not on an edge or corner.
	else
	{
		d_coeff_p = 4*d_coeff;
		active_half[ind_x][ind_y] = d_coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y+1]+passive_half[ind_x][ind_y-1]+passive_half[ind_x+seq][ind_y])/d_coeff_p;
	}
	
}

__global__ void gaussSeidel_RB(double red_matrix_initial[][DIVISIONS], double black_matrix_initial[][DIVISIONS], double red_matrix_final[][DIVISIONS], double black_matrix_final[][DIVISIONS], double d_guess, double d_coeff, double d_A, double d_ds)
{

    // Initialize array of guesses.
    double diff_matrix;
	bool stop = true;
	int iter = 0;

    // Initialize matrices.
	int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
	int ind_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (ind_x < DIVISIONS/2 && ind_y < DIVISIONS)
	{
		red_matrix_initial[ind_x][ind_y] = d_guess;
		red_matrix_final[ind_x][ind_y] = d_guess;
		black_matrix_initial[ind_x][ind_y] = d_guess;
		black_matrix_final[ind_x][ind_y] = d_guess;
	}
	
    while (stop)
    {

		// Lets say red is in position (0,0).
        differencingOperation(red_matrix_final, black_matrix_final, ind_x, ind_y, 1, d_coeff);

		__syncthreads();

		differencingOperation(black_matrix_final, red_matrix_final, ind_x, ind_y , 2, d_coeff);
		

        // Get the absolute value of the difference.  Yeah that'd be great!

		abs(red_matrix_initial[ind_x][ind_y] - red_matrix_final[ind_x][ind_y]) + abs(black_matrix_initial[ind_x][ind_y] - black_matrix_final[ind_x][ind_y]);

		
		
			diff_matrix = 0.;
			for (int k = 0 ; k < DIVISIONS/2 ; k++)
			{
				for (int n = 0; n<DIVISIONS; n++)
				{
					if (ind_x == k && ind_y == n) 
					{
						diff_matrix += abs(red_matrix_initial[k][n] - red_matrix_final[k][n]) + abs(black_matrix_final[k][n] - black_matrix_final[k][n]);
			
					}
				}
			}

		__syncthreads();

		printf("The difference between the matrices is %.4f\n",diff_matrix);

        if (ind_x == 0 && ind_y == 0)
		{
			iter++;
		}

        if ((diff_matrix / (DIVISIONS*DIVISIONS)) < TOLERANCE)
        {
            //printf("The solution converges after %.d iterations.\n",iter);
			printf("That took %.i iterations\n",iter);
			stop = false;
		}

		__syncthreads();
		//printf("This Happened Too!\n");
		red_matrix_initial[ind_x][ind_y] = red_matrix_final[ind_x][ind_y];
		black_matrix_initial[ind_x][ind_y] = black_matrix_final[ind_x][ind_y];
    }
}


int main()
{
	// Get device properties and set threads to be max thread size.  
	// We need the threads to fit the matrix correctly so reject the program if they don't.
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0 );
	int mt = prop.maxThreadsPerBlock;
	cout << "Max Threads Per Block: " << mt << endl;
	int thread = sqrt(mt);

	if (DIVISIONS%(2*thread) != 0)
	{
		printf("Error: DIVISIONS must be a multiple of %.i.  That's twice the thread dimension.\n",(2*thread));
		return 0;
	}

    double ds = LENS/((double)(DIVISIONS-1));
    double A = DZ * ds;
    double guess;
	double coeff = TH_COND * A / ds;
    double bound_cond[4];
	int y_gr = DIVISIONS/thread;
	int x_gr = y_gr/2;
	double *d_red;
	double *d_red2;
	double *d_black; 
	double *d_black2;

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

	// Put the constants in constant memory.
	cudaMemcpyToSymbol( d_bc, bound_cond, sizeof(bound_cond));
	//cudaMemcpyToSymbol( d_A, A, sizeof(A));
	//cudaMemcpyToSymbol( d_guess, guess, sizeof(double));
	//cudaMemcpyToSymbol( d_ds, ds, sizeof(ds));
	//cudaMemcpyToSymbol( d_coeff, coeff, sizeof(coeff));

	// Copy the Initial arrays to the GPU.
	cudaMalloc((void **) &d_red, sizeof(red));
	cudaMalloc((void **) &d_black, sizeof(black));
	cudaMalloc((void **) &d_red2, sizeof(red));
	cudaMalloc((void **) &d_black2, sizeof(black));

	dim3 grids(x_gr,y_gr);
	dim3 threads(thread,thread);

    double wall0 = clock();

    gaussSeidel_RB <<< grids, threads >>> ((double(*) [DIVISIONS]) d_red,(double(*) [DIVISIONS]) d_black,(double(*) [DIVISIONS]) d_red2,(double(*) [DIVISIONS]) d_black2, guess, coeff, A, ds);

	cudaMemcpy(&red,&d_red2,sizeof(red),cudaMemcpyDeviceToHost);
	cudaMemcpy(&black,&d_black2,sizeof(red),cudaMemcpyDeviceToHost);
    double wall1 = clock();
    double timed = (wall1-wall0)/double(CLOCKS_PER_SEC);

    printf("That took %.8f seconds.\n",timed);
	
	cudaDeviceSynchronize();
	// Write it out!
	ofstream filewrite;
	filewrite.open("C:\\Users\\Philadelphia\\Documents\\1_SweptTimeResearch\\GaussSeidel\\GaussSeidelCUDA\\GS_outputCUDA.dat", ios::trunc);
	filewrite << DIVISIONS << "\n" << ds;

    for (int k = 0; k < x_dim; k++)
    {
        for (int n = 0; n < DIVISIONS; n++)
        {
            filewrite << "\n" << red[k][n] << "\n" << black[k][n];
        }
    }

    filewrite.close();

	cudaFree(d_red);
	cudaFree(d_red2);
	cudaFree(d_black);
	cudaFree(d_black2);

    return 0;
}