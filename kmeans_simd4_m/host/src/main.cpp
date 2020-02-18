#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "AOCLUtils/kMeans.h"
#include <time.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <CL/cl.h>

using namespace aocl_utils;

#pragma GCC diagnostic ignored "-Wunused-result"

#define GROUPSIZE 2000
#define MAX_SOURCE_SIZE (0x10000)

// Runtime constants
// Used to define the work set over which this kernel will execute.
static const size_t work_group_size = 8;
static const int thread_id_to_output = 2;

// OpenCL runtime configuration
static cl_platform_id platformID = NULL;
static cl_device_id deviceID = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_kernel kernel2 = NULL;
static cl_program program = NULL;

//initializing 
cl_int ret;
clock_t start, end;
double cpu_time_used;

// Function prototypes
void cleanup();
void readFiles(float *X, float *Y, char *x, char *y, int n);
void writeFiles(int *C, float *CX, float *CY, int k, int n);

// Entry point.
int main(int argc, char **argv) {

	int n;
	int k;

	char *xFilename;
	char *yFilename;
	printf("Please Enter Value of n \n");
	scanf("%d", &n);


	if (n == 10000) {
		xFilename = "Data/X10000.txt";
		yFilename = "Data/Y10000.txt";
	}
	else if (n == 100000) {
		xFilename = "Data/X100000.txt";
		yFilename = "Data/Y100000.txt";
	}
	else if (n == 1000000) {
		xFilename = "Data/X1000000.txt";
		yFilename = "Data/Y1000000.txt";
	}
	else {
		printf("Wrong Value of n \n");
		exit(1);
	}

	printf("Please Enter Value of K \n");
	scanf("%d", &k);

	printf("number of points n= %d\n", n);
	printf("number of clusters k= %d\n", k);
	printf("Groupsize = %d\n", GROUPSIZE);

	int numGroups = n / GROUPSIZE;

	float *X = (float*)malloc(n * sizeof(float));
	float *Y = (float*)malloc(n * sizeof(float));
	float *CX = (float*)malloc(k * sizeof(float));
	float *CY = (float*)malloc(k * sizeof(float));
	float *oldX = (float*)malloc(k * sizeof(float));
	float *oldY = (float*)malloc(k * sizeof(float));
	int *counts = (int*)malloc(numGroups * k * sizeof(int));
	int *flag = (int*)malloc(sizeof(int));
	int *clusters = (int*)malloc(n * sizeof(int));
	flag[0] = 0;

	for (int i = 0; i < k; i++)
	{
		float range = 8;
		CX[i] = range * ((float)rand() / (float)RAND_MAX);
		CY[i] = range * ((float)rand() / (float)RAND_MAX);
	}

	readFiles(X, Y, xFilename, yFilename, n);

	if (!setCwdToExeDir()) {
		return false;
	}

	// Get the OpenCL platform.
	platformID = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
	if (platformID == NULL) {
		printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
		return false;
	}
	// Query the available OpenCL devices.
	scoped_array<cl_device_id> devices;
	cl_uint num_devices;

	devices.reset(getDevices(platformID, CL_DEVICE_TYPE_ALL, &num_devices));

	// We'll just use the first device.
	deviceID = devices[0];

	// Create the context.
	context = clCreateContext(NULL, 1, &deviceID, &oclContextCallback, NULL, &ret);
	checkError(ret, "Failed to create context");

	// Create the command queue.
	queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &ret);
	checkError(ret, "Failed to create command queue");

	// Creating Buffers 
	cl_mem xMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), X, &ret);
	cl_mem yMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), Y, &ret);
	cl_mem cxMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * sizeof(float), CX, &ret);
	cl_mem cyMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * sizeof(float), CY, &ret);
	cl_mem oxMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * sizeof(float), oldX, &ret);
	cl_mem oyMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, k * sizeof(float), oldY, &ret);
	cl_mem gcMem = clCreateBuffer(context, CL_MEM_READ_WRITE, numGroups * k * sizeof(int), NULL, &ret);
	cl_mem sxMem = clCreateBuffer(context, CL_MEM_READ_WRITE, numGroups * k * sizeof(int), NULL, &ret);
	cl_mem syMem = clCreateBuffer(context, CL_MEM_READ_WRITE, numGroups * k * sizeof(int), NULL, &ret);
	cl_mem fMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), flag, &ret);
	cl_mem clMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * sizeof(int), clusters, &ret);
	if (!xMem || !yMem || !cxMem || !cyMem || !gcMem || !sxMem || !syMem || !oxMem || !oyMem || !fMem || !clMem)
	{
		printf("Failed to allocate memory\n");
		exit(1);
	}

	// Create the program.
	std::string binary_file = getBoardBinaryFile("kmeans", deviceID);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &deviceID, 1);

	ret = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(ret, "Failed to build program");

	// Create the kernel - name passed in here must match kernel name in the
	// original CL file, that was compiled into an AOCX file using the AOC tool
	const char *kernel_assign = "assignCluster";  // Kernel name, as defined in the CL file
	kernel = clCreateKernel(program, kernel_assign, &ret);
	checkError(ret, "Failed to create kernel");

	const char *kernel_update = "updateMean";  // Kernel name, as defined in the CL file
	kernel2 = clCreateKernel(program, kernel_update, &ret);
	checkError(ret, "Failed to create kernel");

	// Setting arguments in 1st kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&clMem);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&xMem);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&yMem);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&cxMem);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&cyMem);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&sxMem);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&syMem);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&gcMem);
	clSetKernelArg(kernel, 8, GROUPSIZE * sizeof(int), NULL);
	clSetKernelArg(kernel, 9, GROUPSIZE * sizeof(float), NULL);
	clSetKernelArg(kernel, 10, GROUPSIZE * sizeof(float), NULL);
	clSetKernelArg(kernel, 11, sizeof(int), (void *)&n);
	clSetKernelArg(kernel, 12, sizeof(int), (void *)&k);

	// Setting arguments in 2nd kernel
	cl_uint *rets = (cl_uint *)malloc(14 * sizeof(cl_uint));
	rets[0] = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&cxMem);
	rets[1] = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&cyMem);
	rets[2] = clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *)&sxMem);
	rets[3] = clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void *)&syMem);
	rets[4] = clSetKernelArg(kernel2, 4, sizeof(cl_mem), (void *)&oxMem);
	rets[5] = clSetKernelArg(kernel2, 5, sizeof(cl_mem), (void *)&oyMem);
	rets[6] = clSetKernelArg(kernel2, 6, sizeof(cl_mem), (void *)&gcMem);
	rets[7] = clSetKernelArg(kernel2, 7, GROUPSIZE * sizeof(int), NULL);
	rets[8] = clSetKernelArg(kernel2, 8, GROUPSIZE * sizeof(float), NULL);
	rets[9] = clSetKernelArg(kernel2, 9, GROUPSIZE * sizeof(float), NULL);
	rets[10] = clSetKernelArg(kernel2, 10, k * sizeof(int), NULL);
	rets[11] = clSetKernelArg(kernel2, 11, sizeof(int), (void *)&k);
	rets[12] = clSetKernelArg(kernel2, 12, sizeof(int), (void *)&numGroups);
	rets[13] = clSetKernelArg(kernel2, 13, sizeof(cl_mem), (void *)&fMem);
	for (int i = 0; i < 14; i++)
	{
		if (rets[i] != CL_SUCCESS)
		{
			printf("Set argument %d error %d\n", i, rets[i]);
			exit(1);
		}
	}

	printf("\nKernel initialization is complete.\n");
	printf("Launching the kernel...\n\n");


	size_t wgSize;
	size_t gSize;
	int iters = 0;

	// starting time calculation from here
	start = clock();

	// Launch the kernels
	do
	{
		iters++;
		gSize = n;
		wgSize = GROUPSIZE;


		// invoking 1st kernel
		ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gSize, &wgSize, 0, NULL, NULL);
		checkError(ret, "Failed to launch kernel");
		// Wait for command queue to complete pending events
		ret = clFinish(queue);
		checkError(ret, "Failed to finish");

		// global size for 2nd kernel
		gSize = GROUPSIZE;

		// invoking 2nd kernel
		ret = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, &gSize, &wgSize, 0, NULL, NULL);
		checkError(ret, "Failed to launch kernel2");
		ret = clFinish(queue);

		// Reading flag value returning from UpdateMean cluster
		ret = clEnqueueReadBuffer(queue, fMem, CL_TRUE, 0, sizeof(int), flag, 0, NULL, NULL);
		checkError(ret, "1Read Buffer Error");
	} while (flag[0]);

	// Reading final results which are clusters point of X and Y axis
	ret = clEnqueueReadBuffer(queue, cyMem, CL_TRUE, 0, k * sizeof(float), CY, 0, NULL, NULL);
	checkError(ret, "Read Buffer Error cymem");

	ret = clEnqueueReadBuffer(queue, cxMem, CL_TRUE, 0, k * sizeof(float), CX, 0, NULL, NULL);
	checkError(ret, "Read Buffer Error cxMem");

	ret = clEnqueueReadBuffer(queue, clMem, CL_TRUE, 0, n * sizeof(int), clusters, 0, NULL, NULL);
	checkError(ret, "Read Buffer Error clMem");

	// stop the time.
	end = clock();
	printf("\nKernel execution is complete.\n");
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("total seconds: %f\n", cpu_time_used);

	// writing files in the text format
	writeFiles(clusters, CX, CY, k, n);

	// Free the resources allocated
	cleanup();
	return 0;
}

/////// HELPER FUNCTIONS ///////

void readFiles(float *X, float *Y, char *x, char *y, int n)
{
	FILE *xFile = fopen(x, "r");
	FILE *yFile = fopen(y, "r");
	if (xFile == NULL || yFile == NULL)
	{
		printf("Could not read files\n");
		exit(1);
	}
	for (int i = 0; i < n; i++)
	{
		fscanf(xFile, "%f", &X[i]);
		fscanf(yFile, "%f", &Y[i]);
	}

	fclose(xFile);
	fclose(yFile);
}

void writeFiles(int *C, float *CX, float *CY, int k, int n)
{
	FILE *cFile = fopen("../OCLC.txt", "w");
	for (int i = 0; i < n; i++)
	{
		fprintf(cFile, "%d", C[i]);
		if (i < n - 1)
		{
			fprintf(cFile, "\n");
		}
	}

	FILE *cxFile = fopen("../OCLCX.txt", "w");
	for (int i = 0; i < k; i++)
	{
		fprintf(cxFile, "%f", CX[i]);
		if (i < k - 1)
		{
			fprintf(cxFile, "\n");
		}
	}

	FILE *cyFile = fopen("../OCLCY.txt", "w");
	for (int i = 0; i < k; i++)
	{
		fprintf(cyFile, "%f", CY[i]);
		if (i < k - 1)
		{
			fprintf(cyFile, "\n");
		}
	}

	fclose(cFile);
	fclose(cyFile);
	fclose(cxFile);
}

// Free the resources allocated during initialization
void cleanup() {
	if (kernel) {
		clReleaseKernel(kernel);
	}
	if (program) {
		clReleaseProgram(program);
	}
	if (queue) {
		clReleaseCommandQueue(queue);
	}
	if (context) {
		clReleaseContext(context);
	}
}