

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include <thrust\reduce.h>
#include <thrust\execution_policy.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>

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
#define REAL float

__device__ void cornerSource (REAL BC1, REAL BC2, REAL *source, REAL coff)
{
	printf("Corner Source was called!\n");
    if (BC1>0)
    {
        if (BC2>0)
        {
            source[0] = 2*coff*(BC1 + BC2);
            source[1] = -4*coff;
        }
        else
        {
            source[0] = 2*coff*BC1;
            source[1] = -2*coff;
        }
    }
	
    else if (BC2>0)
    {
        source[0] = 2* coff *BC2;
        source[1] = -2 * coff;
    }
    else
    {
        source[0] = 0;
        source[1] = 0;
		
    }
}

__device__ void differencingOperation(REAL *active_half, REAL *passive_half, int ind_x, int ind_y, int id, int turn, REAL *d_const)
{
	
    REAL d_coeff_p;
	int grd = gridDim.x*blockDim.x;
    REAL *source = new REAL[2];
	// Negative seq means active half starts second.  Positive seq means active half starts first.
	int seq = (((turn+ind_y)%2)*2)-1;

	// This is to catch indices outside the bounds.

	// If bottom row.
	if (ind_y == 0)
	{
		// If bottom left (SouthWest) corner and red.
		if (ind_x == 0 && turn == 0)
		{
			cornerSource(d_const[2],d_const[3], source, d_const[5]);
			d_coeff_p = 2*d_const[5]-source[1];	
			active_half[id] = (d_const[5]*(passive_half[id] + passive_half[id+grd])+source[0])/d_coeff_p;
			printf("Sequence is %.f.\n",seq);
			
		}
		// If bottom right (SouthEast) corner and black.
		else if (ind_x == (DIVISIONS-1) && turn == 1)
		{
			cornerSource(d_const[2],d_const[1], source, d_const[5]);
			d_coeff_p = 2*d_const[5]-source[1];
			active_half[id] = (d_const[5]*(passive_half[id] + passive_half[id+grd])+source[0])/d_coeff_p;
			printf("Southeast Happened (turn2)\n");
		}
		// Bottom row no corner.
		else
		{
			// Check South Boundary Condition.  If it's constant temperature:
			if (d_const[2]>0)
			{
				source[0] = 2*d_const[5]*d_const[2];
				source[1] = -2*d_const[5];
				d_coeff_p = 3*d_const[5]-source[1];
				active_half[id] = (d_const[5]*(passive_half[id]+passive_half[id+grd]+passive_half[id+1])+source[0])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_const[5];
				active_half[id] = d_const[5]*(passive_half[id]+passive_half[id+grd]+passive_half[id+1])/d_coeff_p;
				//printf("The active_half has a value %.f at [%.f][%.f].\n",active_half[id],ind_x,ind_y);
			}
		}
	}
	// If top row
	else if (ind_y == DIVISIONS/2-1)
	{
		// If top right (NorthEast) corner and red.
		if (ind_x == (DIVISIONS-1) && turn == 0)
		{
			cornerSource(d_const[0],d_const[1], source, d_const[5]);
			d_coeff_p = 2*d_const[5]-source[1];
			active_half[id] = (d_const[5]*(passive_half[id] + passive_half[id-grd])+source[0])/d_coeff_p;	
			printf("Northeast Happened (turn1)\n");
		}
		// If top left (NorthWest) corner and black.
		else if (ind_x == 0 && turn == 1)
		{
			cornerSource(d_const[0],d_const[3], source, d_const[5]);
			d_coeff_p = 2*d_const[5]-source[1];
			active_half[id] = (d_const[5]*(passive_half[id] + passive_half[id-grd])+source[0])/d_coeff_p;
			printf("Sequence is %.f.\n",seq);
		}
		// Top row no corner.  The top row is the compliment of the bottom row so the operation for seq is reversed.
		else
		{
			// Check North Boundary Condition.  If it's constant temperature:
			if (d_const[0]>0)
			{
				source[0] = 2*d_const[5]*d_const[0];
				source[1] = -2*d_const[5];
				d_coeff_p = 3*d_const[5]-source[1];
				active_half[id] = (d_const[5]*(passive_half[id]+passive_half[id-grd]+passive_half[id+seq])+source[0])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_const[5];
				active_half[id] = d_const[5]*(passive_half[id]+passive_half[id-grd]+passive_half[id+seq])/d_coeff_p;
			}
		}
	}
	// Check side walls.  This is West when the matrix starts the row, that's when seq is -1.
	else if (ind_x == 0 && seq == -1)
	{
		if (d_const[3]>0)
		{
			source[0] = 2*d_const[5]*d_const[3];
			source[1] = -2*d_const[5];
			d_coeff_p = 3*d_const[5]-source[1];
			active_half[id] = (d_const[5]*(passive_half[id]+ passive_half[id+grd] + passive_half[id-grd])+source[0])/d_coeff_p;
		}
		else
		{
			d_coeff_p = 3*d_const[5];
			active_half[id] = d_const[5]*(passive_half[id]+passive_half[id+grd]+passive_half[id-grd])/d_coeff_p;

		}
	}
	
	// This is East when the matrix ends the row.
	else if (ind_x == (DIVISIONS-1) && seq == 1)
	{
		if (d_const[1]>0)
		{
			source[0] = 2*d_const[5]*d_const[1];
			source[1] = -2*d_const[5];
			d_coeff_p = 3*d_const[5]-source[1];
			active_half[id] = (d_const[5]*(passive_half[id]+ passive_half[id+grd] + passive_half[id-grd])+source[0])/d_coeff_p;
		}
		else
		{
			d_coeff_p = 3*d_const[5];
			active_half[id] = d_const[5]*(passive_half[id]+passive_half[id+grd]+passive_half[id-grd])/d_coeff_p;
		}
	}
	// Every cell not on an edge or corner.
	else
	{
		d_coeff_p = 4*d_const[5];
		active_half[id] = d_const[5]*(passive_half[id]+passive_half[id+grd]+passive_half[id-grd]+passive_half[id+seq])/d_coeff_p;
	}
	
}

__global__ void gaussSeidel_RB(REAL *red_matrix_initial, REAL *black_matrix_initial, REAL *red_matrix_final, REAL *black_matrix_final, REAL *diff_matrix, REAL *d_con, int sz)
{

    // Initialize array of guesses.
	int iter = 0;
	
    // Initialize matrices.
	int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
	int ind_y = blockIdx.y * blockDim.y + threadIdx.y;
	int id = ind_x+ind_y*DIVISIONS;
	if (ind_x < DIVISIONS && ind_y < DIVISIONS/2)
	{
		red_matrix_initial[id] = d_con[4];
		red_matrix_final[id]  = d_con[4];
		black_matrix_initial[id] = d_con[4];
		black_matrix_final[id] = d_con[4];
	}
	
    while(true)
    {
		// Lets say red is in position (0,0).
        differencingOperation(red_matrix_final, black_matrix_final, ind_x, ind_y, id, 0, d_con);

		if (ind_x == 0 && ind_y == 0)
		{
			printf("The first difference completed\n");
			printf("Black[5000] is %.5f\n",black_matrix_final[5000]);
			printf("Red[5000] is %.5f\n",red_matrix_final[5000]);
			iter++;
			printf("The Constant Memory holds: \n");
			for (int k = 0; k<6; k++) printf("%.8f\n",d_con[k]);

			printf("Iteration: %.i.\n",iter);
		}

		differencingOperation(black_matrix_final, red_matrix_final, ind_x, ind_y, id, 1, d_con);

		if (ind_x == 100 && ind_y == 100)
		{
			printf("The Second difference completed\n");
		}
		__syncthreads();
		
		diff_matrix[id] = abs(red_matrix_initial[id]-red_matrix_final[id]) + abs(black_matrix_initial[id]-black_matrix_initial[id]);
		__syncthreads();

		REAL dm2 = thrust::reduce(thrust::device,diff_matrix ,diff_matrix + sz);
		__syncthreads();
		if (id == 1500)
		{
			printf("The reduction gives %.5f difference \n",dm2);
		}
        if ((dm2 / (REAL)(.5*DIVISIONS*DIVISIONS)) < TOLERANCE)
        {
            printf("The solution converges after %.d iterations.\n",iter);
			return;
		}
		
		red_matrix_initial[id] = red_matrix_final[id];
		black_matrix_initial[id] = black_matrix_final[id];
		__syncthreads();
    }
}

int main()
{

	cudaDeviceSynchronize();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	// Get device properties and set threads to be max thread size.  
	// We need the threads to fit the matrix correctly so reject the program if they don't.
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0 );
	int mt = prop.maxThreadsPerBlock;
	cout << "Max Threads Per Block: " << mt << endl;
	int thread = 32;

	if (DIVISIONS%(2*thread) != 0)
	{
		printf("Error: DIVISIONS must be a multiple of %.i.  That's twice the thread dimension.\n",(2*thread));
		return 0;
	}

	const int sz = .5*DIVISIONS*DIVISIONS;
	REAL *red = new REAL[sz];
	REAL *black = new REAL[sz];
	thrust::host_vector<REAL> temp_c(6);
	REAL ds = LENS/((REAL)(DIVISIONS-1));
    REAL A = DZ * ds;
	int x_gr = DIVISIONS/thread;
	int y_gr = x_gr/2;
	cout << "Grid Dims: Threads: " << thread <<  " x = " << x_gr << " y = " << y_gr << endl;
	REAL *d_red;
	REAL *d_red2;
	REAL *d_black; 
	REAL *d_black2;
	
    // Get initial conditions
 //   cout << "Provide Boundary conditions for each edge of the slab.\nEnter Constant Temperature in KELVIN\nor a negative number for an insulated boundary:\nNorth: \n";
	//cin >> temp_c[0];
 //   cout << "East: \n";
 //   cin >> temp_c[1];
 //   cout << "South: \n";
 //   cin >> temp_c[2];
 //   cout << "West: \n";
 //   cin >> temp_c[3];

 //   // Get Guess for slab temperature
 //   cout << "Provide a guess Temperature for the slab in Kelvin:\n";
 //   cin >> temp_c[4];

	// For debugging:
	temp_c[0] = 500.;
	temp_c[1] = 740.;
	temp_c[2] = -9.;
	temp_c[3] = -9.;
	temp_c[4] = 580.;
	temp_c[5] = TH_COND * A / ds;

	// Put the constants in constant memory.
	
	// Copy the Initial arrays to the GPU.
	cudaMalloc((void **) &d_red, sizeof(REAL)*sz);
	cudaMalloc((void **) &d_red2, sizeof(REAL)*sz);
	cudaMalloc((void **) &d_black, sizeof(REAL)*sz);
	cudaMalloc((void **) &d_black2, sizeof(REAL)*sz);
	thrust::device_vector<REAL> h_diff(sz);
	REAL *diff_matrix = thrust::raw_pointer_cast(h_diff.data());

	thrust::device_vector<REAL> t_2 = temp_c;
	REAL *d_const = thrust::raw_pointer_cast(t_2.data());
	//cudaMemcpy(d_const,temp_c,sizeof(temp_c),cudaMemcpyHostToDevice);
	
/*	{
		cudaFree(d_red);
		cudaFree(d_red2);
		cudaFree(d_black);
		cudaFree(d_black2);
		cudaFree(d_const);
		free(temp_c);
		cout << "The Memcpy failed:\n" << endl;
		return 0;
	}*/


	dim3 grids(x_gr,y_gr);
	dim3 threads(thread,thread);

    cudaEventRecord(start,0);

    gaussSeidel_RB <<< grids, threads >>> ( d_red, d_black, d_red2, d_black2, diff_matrix, d_const, sz);

	cudaMemcpy(red,&d_red2,sizeof(red),cudaMemcpyDeviceToHost);
	cudaMemcpy(black,&d_black2,sizeof(red),cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	REAL timed;
	cudaEventElapsedTime(&timed,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    printf("That took %.8f seconds.\n",timed);
	cout << red[5] << endl;
	
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
	//cudaFree(d_const);
	//free(temp_c);

    return 0;
}