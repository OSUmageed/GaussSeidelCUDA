//We'll call this the Stendhal Red Black Scheme.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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

// Define constants for GPU constant memory.
__constant__ double d_bc[4];
__constant__ double d_guess;
__constant__ double d_A;
__constant__ double d_ds;
__constant__ double d_coeff;
__constant__ double d_numel;

__device__ void *cornerSource (int BC1, int BC2, double *source)
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

__device__ void differencingOperation(double *active_half, double *passive_half, int active_index, int turn)
{
	
    double d_coeff_p;
    double *source = new double[2];
	// Negative seq means active half starts second.  Positive seq means active half starts first.
	int seq = pow(-1,turn+blockIdx.x);

	// If bottom row.
	if (active_index < blockDim.x)
	{
		// If bottom left (SouthWest) corner and red.
		if (active_index == 0 && turn == 1)
		{
			cornerSource(d_bc[2],d_bc[3], source);
			d_coeff_p = 2*d_coeff-source[1];
			active_half[active_index] = (d_coeff*(passive_half[active_index]+passive_half[blockDim.x])+source[0])/d_coeff_p;	
		}
		// If bottom right (SouthEast) corner and black.
		else if (active_index == (blockDim.x-1) && turn == 2)
		{
			cornerSource(d_bc[2],d_bc[1], source);
			d_coeff_p = 2*d_coeff-source[1];
			active_half[active_index] = (d_coeff*(passive_half[active_index]+passive_half[active_index+blockDim.x])+source[0])/d_coeff_p;
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
				active_half[active_index] = d_coeff*(passive_half[active_index]+passive_half[active_index+seq]+passive_half[active_index+blockDim.x])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_coeff;
				active_half[active_index] = d_coeff*(passive_half[active_index]+passive_half[active_index+seq]+passive_half[active_index+blockDim.x])/d_coeff_p;

			}
		}
	}
	// If top row
	else if (active_index > d_numel-blockDim.x)
	{
		// If top right (NorthEast) corner and red.
		if (active_index == (d_numel-1) && turn == 1)
		{
			cornerSource(d_bc[0],d_bc[1], source);
			d_coeff_p = 2*d_coeff-source[1];
			active_half[active_index] = (d_coeff*(passive_half[active_index]+passive_half[active_index-blockDim.x])+source[0])/d_coeff_p;	
		}
		// If top left (NorthWest) corner and black.
		else if (active_index == (d_numel-(blockDim.x-1)) && turn == 2)
		{
			cornerSource(d_bc[0],d_bc[3], source);
			d_coeff_p = 2*d_coeff-source[1];
			active_half[active_index] = (d_coeff*(passive_half[active_index]+passive_half[active_index+blockDim.x])+source[0])/d_coeff_p;
		}
		// Top row no corner.
		else
		{
			// Check South Boundary Condition.  If it's constant temperature:
			if (d_bc[2]>0)
			{
				source[0] = 2*d_coeff*d_bc[2];
				source[1] = -2*d_coeff;
				d_coeff_p = 3*d_coeff-source[1];
				active_half[active_index] = d_coeff*(passive_half[active_index]+passive_half[active_index+seq]+passive_half[active_index+blockDim.x])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_coeff;
				active_half[active_index] = d_coeff*(passive_half[active_index]+passive_half[active_index+seq]+passive_half[active_index+blockDim.x])/d_coeff_p;

			}
		}
	}
	// Check side walls.
	else if (j == 0)
	{
		if (i == (rw-1))
		{
			cornerSource(d_bc[0],d_bc[3], source);
			d_coeff_p = 2*d_coeff-source[1];
			active_half[active_index] = (d_coeff*(passive_half[i-1][j] + passive_half[i][j+1])+source[0])/d_coeff_p;
		}
		else
		{
			if (d_bc[3]>0)
			{
				source[0] = 2*d_coeff*d_bc[3];
				source[1] = -2*d_coeff;
				d_coeff_p = 3*d_coeff-source[1];
				active_half[active_index] = (d_coeff*(passive_half[i-1][j] + passive_half[i+1][j] + passive_half[i][j+1])+source[0])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_coeff;
				active_half[active_index] = d_coeff*(passive_half[i+1][j]+passive_half[i-1][j]+passive_half[i][j+1])/d_coeff_p;

			}
		}
	}
	else if (i == (rw-1))
	{
		if (j == (cl-1))
		{
			cornerSource(d_bc[0], d_bc[1], source);
			d_coeff_p = 2*d_coeff-source[1];
			active_half[active_index] = (d_coeff*(passive_half[i-1][j]+passive_half[i][j-1])+source[0])/d_coeff_p;

		}
		else
		{
			if (d_bc[0]>0)
			{
				source[0] = 2*d_coeff*d_bc[0];
				source[1] = -2*d_coeff;
				d_coeff_p = 3*d_coeff-source[1];
				active_half[active_index] = (d_coeff*(passive_half[i-1][j]+passive_half[i][j-1]+passive_half[i][j+1])+source[0])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_coeff;
				active_half[active_index] = d_coeff*(passive_half[i-1][j]+passive_half[i][j-1]+passive_half[i][j+1])/d_coeff_p;

			}
		}
	}
	else if (j == (cl-1))
	{
		if (d_bc[1]>0)
		{
			source[0] = 2*d_coeff*d_bc[1];
			source[1] = -2*d_coeff;
			d_coeff_p = 3*d_coeff-source[1];
			active_half[active_index] = (d_coeff*(passive_half[i-1][j]+passive_half[i+1][j]+passive_half[i][j-1])+source[0])/d_coeff_p;
		}
		else
		{
			d_coeff_p = 3*d_coeff;
			active_half[active_index] = d_coeff*(passive_half[i-1][j]+passive_half[i+1][j]+passive_half[i][j-1])/d_coeff_p;

		}
	}
	else
	{
		d_coeff_p = 4*d_coeff;
		active_half[active_index] = d_coeff*(passive_half[i-1][j]+passive_half[i+1][j]+passive_half[i][j-1]+passive_half[i][j+1])/d_coeff_p;
	}

}


__global__ void gaussSeidel_RB (double *red_matrix_initial, double *black_matrix_initial)
{

    // Initialize array of guesses.
    double diff_matrix;
	double *red_matrix_final;
	double *black_matrix_final;
    int iter = 0;

    // Initialize matrices.
	int indx = blockIdx.x * blockDim.x + threadIdx.x;

	cudaMemcpy(red_matrix_final,red_matrix_initial,sizeof(red_matrix_initial),cudaMemcpyDeviceToDevice);
	cudaMemcpy(black_matrix_final,black_matrix_initial,sizeof(black_matrix_initial),cudaMemcpyDeviceToDevice);

    while (true)
    {

		// Lets say red is in position (0,0).
        differencingOperation(red_matrix_final, black_matrix_final,indx,1);
		__syncthreads();
		differencingOperation(black_matrix_final, red_matrix_final,indx,2);
		__syncthreads();

        // Get the absolute value of the difference.

        diff_matrix = 0;
        for (int k = 0; k < d_numel; k++)
        {

            diff_matrix += abs(red_matrix_initial[k] - red_matrix_final[k]) + abs(black_matrix_initial[k] - black_matrix_final[k]);

        }

        iter++;

        if ((diff_matrix / (d_numel*2)) < TOLERANCE)
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

		// final to initial??
		cudaMemcpy(red_matrix_initial,red_matrix_final,sizeof(red_matrix_initial),cudaMemcpyDeviceToDevice);
		cudaMemcpy(black_matrix_initial,black_matrix_final,sizeof(black_matrix_initial),cudaMemcpyDeviceToDevice);

		__syncthreads();

       /* for (int k = 0; k < rw; k++)
		{

			red_matrix_initial[k] = red_matrix_final[k];
			black_matrix_initial[k] = black_matrix_final[k];
            
        }
		*/
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
	const int numel = DIVISIONS*DIVISIONS/2;
	double red[numel];
    double black[numel];
	double *d_red;
	double *d_black;
	int rw = DIVISIONS;
	int cl = DIVISIONS/2;


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

	for (int k = 0; k<numel; k++)
	{
		red[k] = guess;
		black[k] = guess;
	}

	// Put the constants in constant memory.
	cudaMemcpyToSymbol( d_bc, bound_cond, sizeof(bound_cond));
	cudaMemcpyToSymbol( &d_A, &A, sizeof(A));
	cudaMemcpyToSymbol( &d_ds, &ds, sizeof(ds));
	cudaMemcpyToSymbol( &d_coeff, &coeff, sizeof(coeff));
	cudaMemcpyToSymbol( &d_numel, &numel, sizeof(numel));

	// Copy the Initial arrays to the GPU.
	cudaMemcpy(d_red, &red, sizeof(red), cudaMemcpyHostToDevice);
	cudaMemcpy(d_black, &black, sizeof(black), cudaMemcpyHostToDevice);

    double wall0 = clock();

    gaussSeidel_RB<<<rw,cl>>>(d_red,d_black);
	cudaMemcpy(&red,red_matrix_final,sizeof(red),cudaMemcpyDeviceToHost);
	cudaMemcpy(&black,black_matrix_final,sizeof(black),cudaMemcpyDeviceToHost);

    double wall1 = clock();
    double timed = (wall1-wall0)/double(CLOCKS_PER_SEC);

    printf("That took %.3e seconds.\n",timed);

	cudaFree(&d_A);
	cudaFree(&d_ds);
	cudaFree(&d_guess);
	cudaFree(&d_bc);
	cudaFree(&d_coeff);

    return 0;
}
