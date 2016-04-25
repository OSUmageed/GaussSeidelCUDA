//THE COPY SHOULD WORK IN LINUX!

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
#include <thrust\copy.h>

#include <stdio.h>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <omp.h>

using namespace std;

// Now the plate must be square.  #Divisions should be some multiple of 32, preferably some 2^x.
#define LENS       1.
#define TH_COND    16.
#define DZ         .01
#define DIVISIONS  512.
#define TOLERANCE  1.e-5
#define REAL float

struct absdiff
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // diffmat = (redi-redf)+blacki-blackf)
        thrust::get<4>(t) = fabsf(thrust::get<0>(t) - thrust::get<1>(t)) + fabsf(thrust::get<2>(t) - thrust::get<3>(t));
    }
};

	
REAL *cornerSource(REAL BC1, REAL BC2, REAL coff)
{
	REAL *sc;
	sc = new REAL[2];

    if (BC1>0)
    {
        if (BC2>0)
        {
            sc[0] = 2.0f * coff * (BC1 + BC2);
            sc[1] = 4.0f * coff;
        }
        else
        {
            sc[0] = 2.0f * coff * BC1;
            sc[1] = 2.0f * coff;
        }
    }

    else if (BC2>0)
    {
        sc[0] = 2.0f * coff * BC2;
        sc[1] = 2.0f * coff;
    }
    else
    {
        sc[0] = 0.0f;
        sc[1] = 0.0f;
    }
	return sc;
}

void coefficientFills(REAL *a_ext, REAL *a_int, REAL *temps, REAL a_base, const int turn)
{
	const int grdx = int(DIVISIONS)/2;
	const int ar_len = int(grdx*DIVISIONS);

	//Well at least we get to call this with omp.
	omp_set_num_threads( 8 );

	#pragma omp parallel for default(none),private(sp,k),shared(grdx,a_int,a_ext,temps,a_base,turn,ar_len)
	for (int k = 0; k < ar_len; k++)
	{
		if (k < grdx)
		{
			// If bottom left (SouthWest) corner and red.
			if (k == 0 && turn == 0)
			{
				REAL *sp;
				sp = new REAL[2];
				sp = cornerSource(temps[2],temps[3], a_base);
				a_ext[k] = sp[0];
				a_int[k] = 2.0f * a_base + sp[1];
				free(sp);
			}
			// If bottom right (SouthEast) corner and black.
			else if (k == (grdx-1) && turn == 1)
			{
				REAL *sp;
				sp = new REAL[2];
				sp = cornerSource(temps[2],temps[1], a_base);
				a_ext[k] = sp[0];
				a_int[k] = 2.0f * a_base + sp[1];
				free(sp);
			}
			// Bottom row no corner.
			else
			{
			// Check South Boundary Condition.  If it's constant temperature:
				if (temps[2] > 0)
				{
					a_ext[k] = 2.0f * a_base * temps[2];
					a_int[k] = 5.0f * a_base;
				}
				else
				{
					a_int[k] = 3.0f * a_base;
				}
			}
		}
		// If top row
		else if (k >= (ar_len-grdx))
		{
			// If top right (NorthEast) corner and red.
			if ((k == (ar_len-1)) && turn == 0)
			{
				REAL *sp;
				sp = new REAL[2];
				sp = cornerSource(temps[0],temps[1], a_base);
				a_ext[k] = sp[0];
				a_int[k] = 2.0f * a_base + sp[1];
				free(sp);
			}
			// If top left (NorthWest) corner and black.
			else if ((k == (ar_len-grdx)) && turn == 1)
			{
				REAL *sp;
				sp = new REAL[2];
				sp = cornerSource(temps[0],temps[3], a_base);
				a_ext[k] = sp[0];
				a_int[k] = 2.0f * a_base + sp[1];
				free(sp);
			}
		// Top row no corner.  The top row is the compliment of the bottom row so the operation for seq is reversed.
			else
			{
				// Check North Boundary Condition.  If it's constant temperature:
				if (temps[0]>0)
				{
					a_ext[k] = 2.0f * a_base * temps[0];
					a_int[k] = 5.0f * a_base;
				}
				else
				{
					a_int[k]= 3.0f * a_base;
				}
			}
		}
		// Check side walls.  This is West when the matrix starts the row, that's when seq is -1.
		else if (((k % grdx)== 0) && (((k/grdx + turn) & 1) == 0))
		{
			if (temps[3]>0)
			{
				a_ext[k] = 2.0f * a_base * temps[3];
				a_int[k] = 5.0f * a_base;	
			}
			else
			{				
				a_int[k]= 3.0f * a_base;
			}
		
		}
		// This is East when the matrix ends the row.
		else if (((k % (grdx)) == (grdx-1)) && (((k/grdx + turn) & 1)))
		{
			if (temps[1]>0)
			{
				a_ext[k] = 2.0f * a_base * temps[1];
				a_int[k] = 5.0f * a_base;			
			}
			else
			{				
				a_int[k]= 3.0f * a_base;
			}
			//cout << "East: Turn: " << turn << " Interior: " << a_int[k] << " Exterior: " << a_ext[k] << " Modulo: " << k % grdx << " Row: " << k/grdx << endl;
		}
	// Every cell not on an edge or corner.
		else
		{
			a_int[k] = 4.0f * a_base;
		}
	}
	
}

__global__ void differencingOperation(REAL *active_half, REAL *passive_half, REAL *a_e, REAL *a_i, REAL *ac, const int turn)
{	
	const int grd = int(DIVISIONS)/2;
	int ind_x = blockIdx.x * blockDim.x + threadIdx.x;
	int ind_y = blockIdx.y * blockDim.y + threadIdx.y;
	int id = ind_x + ind_y * grd;

	// Negative seq means active half starts first.  Positive seq means passive half starts first.
	// If it's one it's odd if it's 0 it's even.
	int seq = ((turn + ind_y) & 1) ? 1:-1;

	if (id<(grd*int(DIVISIONS)))
	{
	// If bottom row.
	if (ind_y == 0)
	{
		// If bottom left (SouthWest) corner and red.
		if (ind_x == 0 && turn == 0)
		{
			active_half[id] = (ac[0]*(passive_half[id] + passive_half[id+grd])+a_e[id])/a_i[id];
			
		}
		// If bottom right (SouthEast) corner and black.
		else if (ind_x == (grd-1) && turn == 1)
		{
			active_half[id] = (ac[0]*(passive_half[id] + passive_half[id+grd])+a_e[id])/a_i[id];
			
		}
		// Bottom row no corner.
		else
		{
			active_half[id] = (ac[0]*(passive_half[id]+passive_half[id+grd]+passive_half[id+seq])+a_e[id])/a_i[id];		
			
		}
	}

	// If top row
	else if (ind_y == (int(DIVISIONS)-1))
	{
		// If top right (NorthEast) corner and red.
		if (ind_x == (grd-1) && turn == 0)
		{
			active_half[id] = (ac[0]*(passive_half[id] + passive_half[id-grd])+a_e[id])/a_i[id];
		}
		// If top left (NorthWest) corner and black.
		else if (ind_x == 0 && turn == 1)
		{
			active_half[id] = (ac[0]*(passive_half[id] + passive_half[id-grd])+a_e[id])/a_i[id];
		}
		// Top row no corner.  The top row is the compliment of the bottom row so the operation for seq is reversed.
		else
		{
			active_half[id] = (ac[0]*(passive_half[id]+passive_half[id-grd]+passive_half[id+seq])+a_e[id])/a_i[id];
		}
	}
	// Check side walls.  This is West when the matrix starts the row, that's when seq is -1.
	else if (ind_x == 0 && seq == -1)
	{
		active_half[id] = (ac[0]*(passive_half[id]+ passive_half[id+grd] + passive_half[id-grd])+a_e[id])/a_i[id];
	}
	// This is East when the matrix ends the row.
	else if (ind_x == (grd-1) && seq == 1)
	{
		active_half[id] = (ac[0]*(passive_half[id] + passive_half[id+grd] + passive_half[id-grd])+a_e[id])/a_i[id];
	}
	// Every cell not on an edge or corner.
	else
	{
		active_half[id] = (ac[0]/a_i[id]) * (passive_half[id]+passive_half[id+grd]+passive_half[id-grd]+passive_half[id+seq]);
	}
	}
}

int main()
{

	// Get device properties and set threads to be max thread size.
	// We need the threads to fit the matrix correctly so reject the program if they don't.
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0 );
	int mt = prop.maxThreadsPerBlock;
	int thread = int(sqrtf(float(mt)));

	if (int(DIVISIONS)%(2*thread) != 0)
	{
		printf("Error: DIVISIONS must be a multiple of %.i.  That's twice the thread dimension.\n",(2*thread));
		return 0;
	}

	const int rw = int(DIVISIONS)/2;
	const int sz = rw*int(DIVISIONS);
	
	cout << "Begin!!! \n\n";

	thrust::host_vector<REAL> red(sz);
	thrust::host_vector<REAL> black(sz);
	REAL ds = (REAL)LENS/((REAL)(DIVISIONS-1));
    REAL A = (REAL)DZ * ds;
	const int y_gr = (int)DIVISIONS/thread;
	const int x_gr = y_gr/2;
	REAL dm2;

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

	REAL *ared_caste, *ablack_caste, *ared_casti, *ablack_casti, *a_b, *ahost_red_e, *ahost_black_e, *ahost_red_i, *ahost_black_i;

	ahost_red_e = (REAL *) malloc(sz*sizeof(REAL));
	ahost_black_e = (REAL *) malloc(sz*sizeof(REAL));
	ahost_red_i = (REAL *) malloc(sz*sizeof(REAL));
	ahost_black_i = (REAL *) malloc(sz*sizeof(REAL));

	REAL ab[2] = {(REAL)TH_COND * A / ds, 0};
	REAL temp_c[4];

	// For debugging:
	temp_c[0] = 2.;
	temp_c[1] = 1.;
	temp_c[2] = 1.;
	temp_c[3] = 1.;
	REAL guess = .5;

	// I know that this can get confusing, but in the coefficients (variables starting with a), the e stands for external and the i stands for internal
	// If it helps any, they're always built in the same order, externals first red before black.
	// Set up host vectors and fill them with coefficients.
	for (int k = 0; k<sz; k++)
	{
		ahost_red_e[k] = 0.f;
		ahost_black_e[k] = 0.f;
		ahost_red_i[k] = 0.f;
		ahost_black_i[k] = 0.f;
	}

	coefficientFills(ahost_red_e, ahost_red_i, temp_c, ab[0], 0);
	coefficientFills(ahost_black_e, ahost_black_i, temp_c, ab[0], 1);


	// Copy the Initial arrays to the GPU.
	thrust::device_vector<REAL> d_red_i(sz,guess);
	thrust::device_vector<REAL> d_red_f(sz,guess);
	thrust::device_vector<REAL> d_black_i(sz,guess);
	thrust::device_vector<REAL> d_black_f(sz,guess);	

	// Copy coefficient vectors to device.
	cudaMalloc((void **) &ared_caste, sizeof(REAL)*sz);
	cudaMalloc((void **) &ablack_caste, sizeof(REAL)*sz);
	cudaMalloc((void **) &ared_casti, sizeof(REAL)*sz);
	cudaMalloc((void **) &ablack_casti, sizeof(REAL)*sz);
	cudaMalloc((void **) &a_b, sizeof(REAL)*2);

	// Fill the difference matrix to be reduced as well.
	thrust::device_vector<REAL> diff_mat(sz);

	//Now make all the raw pointers so you can pass them to the kernel.
	REAL *red_cast = thrust::raw_pointer_cast(&d_red_f[0]);
	REAL *black_cast = thrust::raw_pointer_cast(&d_black_f[0]);
	REAL *red_casti = thrust::raw_pointer_cast(&d_red_i[0]);
	REAL *black_casti = thrust::raw_pointer_cast(&d_black_i[0]);

	//The coefficients are vanilla CUDA/C++
	cudaMemcpy(ared_caste, ahost_red_e, sizeof(REAL)*sz, cudaMemcpyHostToDevice);
	cudaMemcpy(ablack_caste, ahost_black_e, sizeof(REAL)*sz, cudaMemcpyHostToDevice);
	cudaMemcpy(ared_casti, ahost_red_i, sizeof(REAL)*sz, cudaMemcpyHostToDevice);
	cudaMemcpy(ablack_casti, ahost_black_i, sizeof(REAL)*sz, cudaMemcpyHostToDevice);
	cudaMemcpy(a_b, ab, sizeof(REAL)*2, cudaMemcpyHostToDevice);

	dim3 grids(x_gr,y_gr);
	dim3 threads(thread,thread);
	bool stops = true;
	int iter = 0;
	double wall0 = clock();

	while (stops)
	{

		differencingOperation <<< grids, threads >>> (red_cast, black_cast, ared_caste, ared_casti, a_b, 0);

		cudaDeviceSynchronize();

		differencingOperation <<< grids, threads >>> (black_cast, red_cast, ablack_caste, ablack_casti, a_b, 1);

		cudaDeviceSynchronize();

		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_red_i.begin(), d_red_f.begin(), d_black_i.begin(), d_black_f.begin(), diff_mat.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_red_i.end(), d_red_f.end(), d_black_i.end(), d_black_f.end(), diff_mat.end())),
			absdiff());


		cudaDeviceSynchronize();

		dm2 = thrust::reduce(diff_mat.begin(),diff_mat.end());

		iter++;

		if (((dm2 /REAL(sz*2)) < TOLERANCE) || (iter>1e7))
		{
			stops = false;
		}
		 
		//d_red_i = d_red_f;

		cudaMemcpy(red_casti, red_cast, sz * sizeof(REAL), cudaMemcpyDeviceToDevice);
		//thrust::copy(d_red_f.begin(), d_red_f.end(), d_red_i.begin());

		//d_black_i = d_black_f;

		cudaMemcpy(black_casti, black_cast, sz * sizeof(REAL), cudaMemcpyDeviceToDevice);
		//thrust::copy(d_black_f.begin(), d_black_f.end(), d_black_i.begin());

		cudaDeviceSynchronize();
		if (iter%200 == 0) 
		{
			cout << "Iteration: " << iter << "  dm:" << dm2/REAL(sz*2) << endl;
			cout << "First red: " << d_red_f[0] << "  Last Black:" << d_black_f[sz-1] << endl;
			cout << "Random red: " << d_red_i[8201] << "  Random Black:" << d_black_i[105] << endl;
		}
	}

    double wall1 = clock();
    double timed = (wall1-wall0)/CLOCKS_PER_SEC;

	printf("Outside the loop\n");

	printf("It converged after %d iterations: \n",iter);

	cout << "That took: " << timed << " seconds" << endl;

	thrust::copy(d_red_f.begin(), d_red_f.end(), red.begin());

	thrust::copy(d_black_f.begin(), d_black_f.end(), black.begin());

	// Write it out!
	ofstream filewrite;
	filewrite.open("C:\\Users\\Philadelphia\\Documents\\1_SweptTimeResearch\\GaussSeidel\\GaussSeidelCUDA\\GS_outputCUDA.dat", ios::trunc);
	filewrite << DIVISIONS << "\n" << ds;

    for (int k = 0; k < sz; k++)
    {
         filewrite << "\n" << red[k] << "\n" << black[k];
    }

    filewrite.close();

	cudaFree(ared_caste);
	cudaFree(ared_casti);
	cudaFree(ablack_caste);
	cudaFree(ablack_casti);
	cudaFree(a_b);
	free(ared_caste);
	free(ared_casti);
	free(ablack_caste);
	free(ablack_casti);
	free(a_b);
	//cudaDeviceReset();
    return 0;

}
