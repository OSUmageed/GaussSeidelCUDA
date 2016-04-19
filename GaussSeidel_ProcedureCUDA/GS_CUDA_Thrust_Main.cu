

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
#include <thrust\for_each.h>
#include <thrust\transform.h>
#include <thrust\iterator\zip_iterator.h>
#include <thrust/copy.h>

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
#define DIVISIONS 512
#define TOLERANCE 1.e-5
#define REAL float

struct absdiff
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // diffmat = (redi-redf)+blacki-blackf) 
        thrust::get<4>(t) = abs(thrust::get<0>(t) - thrust::get<1>(t)) + abs(thrust::get<2>(t) - thrust::get<3>(t));
    }
};

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

__global__ void differencingOperation(REAL *active_half, REAL *passive_half, REAL *d_const, int turn)
{
	int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
	int ind_y = blockIdx.y * blockDim.y + threadIdx.y;
	int id = ind_x+ind_y*DIVISIONS;

    REAL d_coeff_p;
	const int grd = gridDim.x*blockDim.x;
    REAL *source = new REAL[2];

	// Negative seq means active half starts second.  Positive seq means active half starts first.
	int seq = (((turn+ind_y)%2)*2)-1;

	// If bottom row.
	if (ind_y == 0)
	{
		// If bottom left (SouthWest) corner and red.
		if (ind_x == 0 && turn == 0)
		{
			cornerSource(d_const[2],d_const[3], source, d_const[4]);
			d_coeff_p = 2*d_const[4]-source[1];	
			active_half[id] = (d_const[4]*(passive_half[id] + passive_half[id+grd])+source[0])/d_coeff_p;
			printf("Sequence is %.f.\n",seq);
			
		}
		// If bottom right (SouthEast) corner and black.
		else if (ind_x == (DIVISIONS-1) && turn == 1)
		{
			cornerSource(d_const[2],d_const[1], source, d_const[4]);
			d_coeff_p = 2*d_const[4]-source[1];
			active_half[id] = (d_const[4]*(passive_half[id] + passive_half[id+grd])+source[0])/d_coeff_p;
			printf("Southeast Happened (turn2)\n");
		}
		// Bottom row no corner.
		else
		{
			// Check South Boundary Condition.  If it's constant temperature:
			if (d_const[2]>0)
			{
				source[0] = 2*d_const[4]*d_const[2];
				source[1] = -2*d_const[4];
				d_coeff_p = 3*d_const[4]-source[1];
				active_half[id] = (d_const[4]*(passive_half[id]+passive_half[id+grd]+passive_half[id+1])+source[0])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_const[4];
				active_half[id] = d_const[4]*(passive_half[id]+passive_half[id+grd]+passive_half[id+1])/d_coeff_p;
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
			cornerSource(d_const[0],d_const[1], source, d_const[4]);
			d_coeff_p = 2*d_const[4]-source[1];
			active_half[id] = (d_const[4]*(passive_half[id] + passive_half[id-grd])+source[0])/d_coeff_p;	
			printf("Northeast Happened (turn1)\n");
		}
		// If top left (NorthWest) corner and black.
		else if (ind_x == 0 && turn == 1)
		{
			cornerSource(d_const[0],d_const[3], source, d_const[4]);
			d_coeff_p = 2*d_const[4]-source[1];
			active_half[id] = (d_const[4]*(passive_half[id] + passive_half[id-grd])+source[0])/d_coeff_p;
			printf("Sequence is %.f.\n",seq);
		}
		// Top row no corner.  The top row is the compliment of the bottom row so the operation for seq is reversed.
		else
		{
			// Check North Boundary Condition.  If it's constant temperature:
			if (d_const[0]>0)
			{
				source[0] = 2*d_const[4]*d_const[0];
				source[1] = -2*d_const[4];
				d_coeff_p = 3*d_const[4]-source[1];
				active_half[id] = (d_const[4]*(passive_half[id]+passive_half[id-grd]+passive_half[id+seq])+source[0])/d_coeff_p;
			}
			else
			{
				d_coeff_p = 3*d_const[4];
				active_half[id] = d_const[4]*(passive_half[id]+passive_half[id-grd]+passive_half[id+seq])/d_coeff_p;
			}
		}
	}
	// Check side walls.  This is West when the matrix starts the row, that's when seq is -1.
	else if (ind_x == 0 && seq == -1)
	{
		if (d_const[3]>0)
		{
			source[0] = 2*d_const[4]*d_const[3];
			source[1] = -2*d_const[4];
			d_coeff_p = 3*d_const[4]-source[1];
			active_half[id] = (d_const[4]*(passive_half[id]+ passive_half[id+grd] + passive_half[id-grd])+source[0])/d_coeff_p;
		}
		else
		{
			d_coeff_p = 3*d_const[4];
			active_half[id] = d_const[4]*(passive_half[id]+passive_half[id+grd]+passive_half[id-grd])/d_coeff_p;

		}
	}
	
	// This is East when the matrix ends the row.
	else if (ind_x == (DIVISIONS-1) && seq == 1)
	{
		if (d_const[1]>0)
		{
			source[0] = 2*d_const[4]*d_const[1];
			source[1] = -2*d_const[4];
			d_coeff_p = 3*d_const[4]-source[1];
			active_half[id] = (d_const[4]*(passive_half[id]+ passive_half[id+grd] + passive_half[id-grd])+source[0])/d_coeff_p;
		}
		else
		{
			d_coeff_p = 3*d_const[4];
			active_half[id] = d_const[4]*(passive_half[id]+passive_half[id+grd]+passive_half[id-grd])/d_coeff_p;
		}
	}
	// Every cell not on an edge or corner.
	else
	{
		d_coeff_p = 4*d_const[4];
		active_half[id] = d_const[4]*(passive_half[id]+passive_half[id+grd]+passive_half[id-grd]+passive_half[id+seq])/d_coeff_p;
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
	thrust::host_vector<REAL> red(sz);
	thrust::host_vector<REAL> black(sz);
	thrust::host_vector<REAL> temp_c(5);
	REAL ds = LENS/((REAL)(DIVISIONS-1));
    REAL A = DZ * ds;
	int x_gr = DIVISIONS/thread;
	int y_gr = x_gr/2;
	
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
	REAL guess = 580.;
	temp_c[4] = TH_COND * A / ds;

	// Put the constants in constant memory.
	
	// Copy the Initial arrays to the GPU.
	thrust::device_vector<REAL> d_red_i(sz,guess);
	thrust::device_vector<REAL> d_red_f (sz,guess);
	thrust::device_vector<REAL> d_black_i(sz,guess);
	thrust::device_vector<REAL> d_black_f (sz,guess);
	thrust::device_vector<REAL> diff_mat(sz);
	thrust::device_vector<REAL> t_2 = temp_c;

	REAL *d_const = thrust::raw_pointer_cast(&t_2[0]);
	REAL *red_cast = thrust::raw_pointer_cast(&d_red_f[0]);
	REAL *black_cast = thrust::raw_pointer_cast(&d_black_f[0]);

	dim3 grids(x_gr,y_gr);
	dim3 threads(thread,thread);

    cudaEventRecord(start,0);
	bool stops = true;
	int iter = 0;

	while (stops)
	{

		//differencingOperation <<< grids, threads >>> (red_cast, black_cast, d_const, 0);

		int mid;
		//cudaDeviceSynchronize();

		//differencingOperation <<< grids, threads >>> (black_cast, red_cast, d_const, 1);

		int maybe;

		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_red_i.begin(), d_red_f.begin(), d_black_i.begin(), d_black_f.begin(), diff_mat.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_red_i.begin(), d_red_f.begin(), d_black_i.begin(), d_black_f.begin(), diff_mat.begin())),
			absdiff());

		REAL dm2 = thrust::reduce(diff_mat.begin(),diff_mat.end());

		if (dm2 /(sz) <TOLERANCE || iter>1e7)
		{
			stops = false;
		}	

		iter ++;

		thrust::copy(d_red_f.begin(), d_red_f.end(), d_red_i.begin());
		thrust::copy(d_black_f.begin(), d_black_f.end(), d_black_i.begin());
		printf("dm2 is %.f\n",dm2);

	}

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	REAL timed;
	cudaEventElapsedTime(&timed,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("It converged after %.f iterations: \n",iter);
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

    return 0;
}