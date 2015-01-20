/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */




#ifdef __APPLE__

#include <GLUT/glut.h>


#else

#include <GL/freeglut.h>

#endif

#define DEBUG

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "mesh_handler.h"

int THREADS_PER_BLOCK = 0;
int BLOCKS = 0;

dim3 windowSize(1024,512);

double zoomFactor = 0.1;
double draggingSpeed = 2.0;
unsigned int iterations = 255;
int ppEffect = -1;

static GLuint pbo_buffer = 0;
struct cudaGraphicsResource *cuda_pbo_resource;

double start;
double dtime;

int display_mode = 2;

ObjMesh mesh;

uchar4* d_tex = NULL;
GLuint	id_tex;

uchar4* d_dst;
float* h_dst;

float angle = 0;
float t_x = 0;
float t_z = 0;

float f_x = 0;
float f_z = 0;
float f_r = 0;

#define BUFFER_DATA(i) ((char *)0 + i)

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }



__global__ void mandelbrot(uchar4* d_image, int image_width, int image_height) // TODO image size
{

	unsigned long i = blockIdx.x*blockDim.x+threadIdx.x;

	if((i <= image_height*image_width))
	{

		unsigned long y = 	i/(unsigned long)image_width;
		unsigned long x =	i % image_width;

		double c_real = -2.0 + 4.0/image_width  * x ;
		double c_imag = -2.0 + 4.0/image_height * y ;

		int max;
		int count;
		double z_real;
		double z_imag; double temp,lengthsq; z_real=z_imag=0.0; count=0;
		max= 255;

		do
		{
			temp=z_real*z_real - z_imag*z_imag + c_real;
			z_imag = 2*z_real*z_imag + c_imag; z_real=temp;
			lengthsq = z_real*z_real + z_imag*z_imag; count++;
		} while ((lengthsq < 4.0) && (count < max));

		uchar4 color;
		color.x = count * 3;
		color.y = count * 5;
		color.z = count * 7;
		d_image[x+y*image_width] = color;

		__syncthreads();
	}
}





//o~--------------------------------------------------------------------~o//
void initGL(void)
//o~--------------------------------------------------------------------~o//
{

  d_tex = new uchar4[windowSize.x*windowSize.y];
  glEnable(GL_TEXTURE_2D);

  glGenTextures(1,&id_tex);
  glBindTexture(GL_TEXTURE_2D, id_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowSize.x, windowSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, d_tex);

  //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowSize.x, windowSize.y, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));

  glGenBuffers(1, &pbo_buffer);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_buffer);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,
                 sizeof(uchar4) * windowSize.x * windowSize.y,
                 0, GL_STREAM_COPY);

  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard);

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glDepthFunc(GL_LEQUAL);
  glDepthRange(0.0f,1.0f);
  //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

}


//o~--------------------------------------------------------------------~o//
void init_glut(int* argc, char** argv)
//o~--------------------------------------------------------------------~o//
{
	  glutInit(argc, argv);
	  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA);
	  glutInitWindowSize(windowSize.x, windowSize.y);
	  glutCreateWindow("Mandelbrot");

	  //glewInit();
}

//o~--------------------------------------------------------------------~o//
void destroyGL(void)
//o~--------------------------------------------------------------------~o//
{
  cudaGraphicsUnregisterResource(cuda_pbo_resource);

  glDeleteBuffers(1, &pbo_buffer);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}


void display_elizabeth(void)
{


	size_t num_bytes;

	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	cudaGraphicsMapResources(1, &cuda_pbo_resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void **)&d_tex, &num_bytes, cuda_pbo_resource);

	mandelbrot<<<BLOCKS, THREADS_PER_BLOCK>>>(d_tex, windowSize.x, windowSize.y);
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, NULL);

	t_x += f_x;
	t_z += f_z;
	angle += f_r;
	drawMesh(mesh, id_tex, pbo_buffer, windowSize.x, windowSize.y, t_x, t_z, angle);


	glutSwapBuffers();
}



void changeSize(int w, int h) {
// ---
	// Prevent a divide by zero, when window is too short
	if (h == 0)
		h = 1;

	float ratio =  w * 1.0 / h;

	// Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);
	// Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45,ratio,0,1000);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);

}


//o~--------------------------------------------------------------------~o//
void inputCBKeyboard(unsigned char key, int /*x*/, int /*y*/)
//o~--------------------------------------------------------------------~o//
{
  switch (key)
  {
    case 27:
    case 'q':
    case 'Q':
      destroyGL();
      glutDestroyWindow(glutGetWindow());
      return;


	case 'a' :
		f_r = 0.25f; break;
	case 'd' :
		f_r = -0.25f; break;
	case 's' :
		f_r = 0.0f; break;

    default:
      break;
  }
}


void processSpecialKeys(int key, int x, int y) {


	switch(key) {
		case GLUT_KEY_UP :
			f_z = -0.01; break;
		case GLUT_KEY_DOWN :
			f_z = +0.01; break;
		case GLUT_KEY_RIGHT :
			f_x = 0.01; break;
		case GLUT_KEY_LEFT :
			f_x = -0.01; break;
	}
}


void releaseSpecialKey(int key, int x, int y)
{
	switch (key) {
		case GLUT_KEY_UP : f_z = 0.0; break;
		case GLUT_KEY_DOWN : f_z = 0.0; break;
		case GLUT_KEY_RIGHT : f_x = 0.0; break;
		case GLUT_KEY_LEFT : f_x = 0.0; break;
	}
}


void init_lightning()
{

	GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f};
	GLfloat ambientLight[] = { 0.6f, 0.6f, 0.6f, 1.0f };
	GLfloat diffuseLight[] = { 1.0f, 1.0f, 1.0f, 1.0f};

	GLfloat light_position[] = { 20.0, 1.0, 15.0, 0.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT,ambientLight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,diffuseLight);
	glLightfv(GL_LIGHT0,GL_SPECULAR,specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

}


int main(int argc, char **argv) {


	int nDevices;
	cudaGetDeviceCount(&nDevices);


	for (int i = 0; i < nDevices; i++)
	{
		    cudaDeviceProp prop;
		    cudaGetDeviceProperties(&prop, i);
		    printf("  Device Number: %d\n", i);
		    printf("  Device name: %s\n", prop.name);
		    printf("  Memory Clock Rate (KHz): %d\n",
		           prop.memoryClockRate);
		    printf("  Memory Bus Width (bits): %d\n",
		           prop.memoryBusWidth);
		    printf("  Peak Memory Bandwidth (GB/s): %f\n",
		           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		    printf("  Max Threads Per Block: %d\n\n", prop.maxThreadsPerBlock);
		    THREADS_PER_BLOCK = prop.maxThreadsPerBlock;
	}

	BLOCKS = ceil((float)(windowSize.x*windowSize.y)/THREADS_PER_BLOCK);


	init_glut(&argc, argv);
	initGL();


	init_lightning();

	mesh = LoadObjMesh("Elizabeth/Elizabeth.obj");


	glutDisplayFunc(display_elizabeth);
	glutReshapeFunc(changeSize);
	glutIdleFunc(display_elizabeth);

	glutSpecialFunc(processSpecialKeys);
	glutSpecialUpFunc(releaseSpecialKey);
	glutKeyboardFunc(inputCBKeyboard);


	glutMainLoop();


	return 1;
}
