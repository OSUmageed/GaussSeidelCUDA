//We'll call this the Stendhal Red Black Scheme.

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda_runtime_api.h"

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

// Define struct for GPU constant memory.
struct simKnowns{

	double bc[4];
	double guess;
	double A;
	double ds;
	double coeff;
	
};

__constant__ simKnowns d_const;

const int x_dim = DIVISIONS/2;
double red[x_dim][DIVISIONS];
double black[x_dim][DIVISIONS];

__device__ void cornerSource (int BC1, int BC2, double *source)
{
    if (BC1>0)
    {
        if (BC2>0)
        {
            source[0] = 2*d_const.coeff*(BC1 + BC2);
            source[1] = -4*d_const.coeff;
        }
        else
        {
            source[0] = 2*d_const.coeff*BC1;
            source[1] = -2*d_const.coeff;
        }
    }
    else if (BC2>0)
    {
        source[0] = 2*d_const.coeff*BC2;
        source[1] = -2*d_const.coeff;
    }
    else
    {
        source[0] = 0;
        source[1] = 0;
    }
}

__device__ void differencingOperation(double active_half[][DIVISIONS], double passive_half[][DIVISIONS], int ind_x, int ind_y, int turn)
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
			cornerSource(d_const.bc[2],d_const.bc[3], source);
			d_coeff_p = 2*d_const.coeff-source[1];	
			active_half[ind_x][ind_y] = (d_const.coeff*(passive_half[ind_x][ind_y] + passive_half[ind_x][ind_y+1])+source[0])/d_coeff_p;
		}
		// If bottom right (SouthEast) corner and black.
		else if (ind_x == (DIVISIONS/2-1) && turn == 2)
		{
			cornerSource(d_const.bc[2],d_const.bc[1], source);
			d_coeff_p = 2*d_const.coeff-source[1];
			active_half[ind_x][ind_y] = (d_const.coeff*(passive_half[ind_x][ind_y] + passive_half[ind_x][ind_y+1])+source[0])/d_coeff_p;
		}
		// Bottom row no corner.
		else
		{
			// Check South Boundary Condition.  If it's constant temperature:
			if (d_const.bc[2]>0)
			{
				source[0] = 2*d_const.coeff*d_const.bc[2];
				source[1] = -2*d_const.coeff;
				d_coeff_p = 3*d_const.coeff-source[1];
				active_half[ind_x][ind_y] = (d_const.coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y+1]+passive_half[ind_x+seq][ind_y])+source[0])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_const.coeff;
				active_half[ind_x][ind_y] = d_const.coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y+1]+passive_half[ind_x+seq][ind_y])/d_coeff_p;

			}
		}
	}
	// If top row
	else if (ind_y == DIVISIONS-1)
	{
		// If top right (NorthEast) corner and red.
		if (ind_x == (DIVISIONS/2-1) && turn == 1)
		{
			cornerSource(d_const.bc[0],d_const.bc[1], source);
			d_coeff_p = 2*d_const.coeff-source[1];
			active_half[ind_x][ind_y] = (d_const.coeff*(passive_half[ind_x][ind_y] + passive_half[ind_x][ind_y-1])+source[0])/d_coeff_p;	
		}
		// If top left (NorthWest) corner and black.
		else if (ind_x == 0 && turn == 2)
		{
			cornerSource(d_const.bc[0],d_const.bc[3], source);
			d_coeff_p = 2*d_const.coeff-source[1];
			active_half[ind_x][ind_y] = (d_const.coeff*(passive_half[ind_x][ind_y] + passive_half[ind_x][ind_y-1])+source[0])/d_coeff_p;
		}
		// Top row no corner.  The top row is the compliment of the bottom row so the operation for seq is reversed.
		else
		{
			// Check North Boundary Condition.  If it's constant temperature:
			if (d_const.bc[0]>0)
			{
				source[0] = 2*d_const.coeff*d_const.bc[0];
				source[1] = -2*d_const.coeff;
				d_coeff_p = 3*d_const.coeff-source[1];
				active_half[ind_x][ind_y] = (d_const.coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y-1]+passive_half[ind_x-seq][ind_y])+source[0])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_const.coeff;
				active_half[ind_x][ind_y] = d_const.coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y-1]+passive_half[ind_x-seq][ind_y])/d_coeff_p;
			}
		}
	}
	// Check side walls.  This is West when the matrix starts the row, that's when seq is -1.
	else if (ind_x == 0 && seq == -1)
	{
		if (d_const.bc[3]>0)
		{
			source[0] = 2*d_const.coeff*d_const.bc[3];
			source[1] = -2*d_const.coeff;
			d_coeff_p = 3*d_const.coeff-source[1];
			active_half[ind_x][ind_y] = (d_const.coeff*(passive_half[ind_x][ind_y]+ passive_half[ind_x][ind_y+1] + passive_half[ind_x][ind_y-1])+source[0])/d_coeff_p;
		}
		else
		{
			d_coeff_p = 3*d_const.coeff;
			active_half[ind_x][ind_y] = d_const.coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y+1]+passive_half[ind_x][ind_y-1])/d_coeff_p;

		}
	}
	
	// This is East when the matrix ends the row.
	else if (ind_x == (DIVISIONS/2-1) && seq == 1)
	{
		if (d_const.bc[1]>0)
		{
			source[0] = 2*d_const.coeff*d_const.bc[1];
			source[1] = -2*d_const.coeff;
			d_coeff_p = 3*d_const.coeff-source[1];
			active_half[ind_x][ind_y] = (d_const.coeff*(passive_half[ind_x][ind_y]+ passive_half[ind_x][ind_y+1] + passive_half[ind_x][ind_y-1])+source[0])/d_coeff_p;
		}
		else
		{
			d_coeff_p = 3*d_const.coeff;
			active_half[ind_x][ind_y] = d_const.coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y+1]+passive_half[ind_x][ind_y-1])/d_coeff_p;
		}
	}
	// Every cell not on an edge or corner.
	else
	{
		d_coeff_p = 4*d_const.coeff;
		active_half[ind_x][ind_y] = d_const.coeff*(passive_half[ind_x][ind_y]+passive_half[ind_x][ind_y+1]+passive_half[ind_x][ind_y-1]+passive_half[ind_x+seq][ind_y])/d_coeff_p;
	}
	
}

__global__ void gaussSeidel_RB(double red_matrix_initial[][DIVISIONS], double black_matrix_initial[][DIVISIONS], double red_matrix_final[][DIVISIONS], double black_matrix_final[][DIVISIONS])
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
		red_matrix_initial[ind_x][ind_y] = d_const.guess;
		red_matrix_final[ind_x][ind_y] = d_const.guess;
		black_matrix_initial[ind_x][ind_y] = d_const.guess;
		black_matrix_final[ind_x][ind_y] = d_const.guess;
	}
	
    while (stop)
    {

		// Lets say red is in position (0,0).
        differencingOperation(red_matrix_final, black_matrix_final, ind_x, ind_y, 1);

		__syncthreads();

		differencingOperation(black_matrix_final, red_matrix_final, ind_x, ind_y , 2);
		
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
					//printf("The difference between the matrices is %.4f. [%.f][%.f]\n",diff_matrix,k,n);
				}
			}
		}

		__syncthreads();

        if (ind_x == 0 && ind_y == 0)
		{
			iter++;
			printf("Iteration: %.f.\n",iter);
		}

        if ((diff_matrix / (DIVISIONS*DIVISIONS)) < TOLERANCE)
        {
            //printf("The solution converges after %.d iterations.\n",iter);
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

	simKnowns temp_dev; 
	
	temp_dev.ds = LENS/((double)(DIVISIONS-1));
    temp_dev.A = DZ * temp_dev.ds;
	temp_dev.coeff = TH_COND * temp_dev.A / temp_dev.ds;
	int y_gr = DIVISIONS/thread;
	int x_gr = y_gr/2;
	double *d_red;
	double *d_red2;
	double *d_black; 
	double *d_black2;

    // Get initial conditions
    cout << "Provide Boundary conditions for each edge of the slab.\nEnter Constant Temperature in KELVIN\nor a negative number for an insulated boundary:\nNorth: \n";
	cin >> temp_dev.bc[0];
    cout << "East: \n";
    cin >> temp_dev.bc[1];
    cout << "South: \n";
    cin >> temp_dev.bc[2];
    cout << "West: \n";
    cin >> temp_dev.bc[3];

    // Get Guess for slab temperature
    cout << "Provide a guess Temperature for the slab in Kelvin:\n";
    cin >> temp_dev.guess;

	// Put the constants in constant memory.
	cudaMemcpyToSymbol( d_const, &temp_dev, sizeof(temp_dev));
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

    gaussSeidel_RB <<< grids, threads >>> ((double(*) [DIVISIONS]) d_red,(double(*) [DIVISIONS]) d_black,(double(*) [DIVISIONS]) d_red2,(double(*) [DIVISIONS]) d_black2);

	cudaMemcpy(&red,&d_red2,sizeof(red),cudaMemcpyDeviceToHost);
	cudaMemcpy(&black,&d_black2,sizeof(red),cudaMemcpyDeviceToHost);
    double wall1 = clock();
    double timed = (wall1-wall0)/double(CLOCKS_PER_SEC);

    printf("That took %.8f seconds.\n",timed);
	cout << red[5][5] << endl;
	
	cudaDeviceSynchronize();
	// Write it out!
	/*ofstream filewrite;
	filewrite.open("C:\\Users\\Philadelphia\\Documents\\1_SweptTimeResearch\\GaussSeidel\\GaussSeidelCUDA\\GS_outputCUDA.dat", ios::trunc);
	filewrite << DIVISIONS << "\n" << ds;

    for (int k = 0; k < x_dim; k++)
    {
        for (int n = 0; n < DIVISIONS; n++)
        {
            filewrite << "\n" << red[k][n] << "\n" << black[k][n];
        }
    }*/

    // filewrite.close();

	cudaFree(d_red);
	cudaFree(d_red2);
	cudaFree(d_black);
	cudaFree(d_black2);

    return 0;
}