/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

#include <cuda_gl_interop.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>




float angle = 0.0f;

int image_height = 900;
int image_width = 1200;

float mesh_x = 0;
float mesh_z = -0.5;

float f_x = 0;
float f_z = 0;

float rotation_speed = 0.0f;			// rotating mesh
float offset 		= 0.0f;				// offset of texture

//unsigned int image[image_width][image_height];

// Starting position and scale
double xOff = -0.5;
double yOff = 0.0;
double scale = 3.2;

#define BUFFER_DATA(i) ((char *)0 + i)

// Starting animation frame and anti-aliasing pass
int animationFrame = 0;
int animationStep = 0;
int pass = 0;

//Source image on the host side
uchar4 *h_Src = 0;

// Destination image on the GPU side
uchar4* d_dst = NULL;

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;

struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

// gl_Shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";


int THREADS_PER_BLOCK = 0;
int BLOCKS = 0;


cudaGraphicsResource_t cudaTexRes;



#define TOKEN_VERTEX_POS "v"
#define TOKEN_VERTEX_NOR "vn"
#define TOKEN_VERTEX_TEX "vt"
#define TOKEN_FACE "f"

struct Vector2f{
    float x, y;
};
struct Vector3f{
    float x, y, z;
};

struct ObjMeshVertex{
    Vector3f pos;
    Vector2f texcoord;
    Vector3f normal;
};

/* This is a triangle, that we can render */
struct ObjMeshFace{
    ObjMeshVertex vertices[3];
};

/* This contains a list of triangles */
struct ObjMesh{
    std::vector<ObjMeshFace> faces;
};

/* Internal structure */
struct _ObjMeshFaceIndex
{
    int pos_index[3];
    int tex_index[3];
    int nor_index[3];
};

ObjMesh mesh;

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



__global__ void mandelbrot(uchar4* d_image, int image_width, int image_height, float off) // TODO image size
{

	unsigned long i = blockIdx.x*blockDim.x+threadIdx.x;

	if((i <= image_height*image_width))
	{

		unsigned long y = 	i/(unsigned long)image_width;
		unsigned long x =	i % image_width;

		double c_real = -2.0 + 4.0/image_width  * (x / off);
		double c_imag = -2.0 + 4.0/image_height * (y / off);

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


void runCUDA(bool bUseOpenGL, bool fp64, int mode)
{


	// DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_dst, gl_PBO));
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource);

	float real_off = (1.3+sin(offset));
	mandelbrot<<<BLOCKS,THREADS_PER_BLOCK>>>(d_dst, image_width, image_height, real_off);
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);


}




GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}

void initOpenGLBuffers(int w, int h)
{
    // delete old buffers
    if (h_Src)
    {
        free(h_Src);
        h_Src = 0;
    }

    if (gl_Tex)
    {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }

    if (gl_PBO)
    {
        //DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    // check for minimized window
    if ((w==0) && (h==0))
    {
        return;
    }

    // allocate new buffers
    h_Src = (uchar4 *)malloc(w * h * 4);

    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
    //While a PBO is registered to CUDA, it can't be used
    //as the destination for OpenGL drawing calls.
    //But in our particular case OpenGL is only used
    //to display the content of the PBO, specified by CUDA kernels,
    //so we need to register/unregister it only once.

    // DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard);
    printf("PBO created.\n");

    // load shader program
    gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void cleanup()
{
    if (h_Src)
    {
        free(h_Src);
        h_Src = 0;
    }


    //DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);
}


void changeSize(int w, int h) {

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

	initOpenGLBuffers(w, h);
	image_width = w;
	image_height = h;
	pass = 0;

}





void drawMesh()
{
    glPushMatrix();
    mesh_x = mesh_x + f_x;
    mesh_z = mesh_z + f_z;
    glTranslatef(mesh_x,-0.2	,mesh_z);
    //glColor3f(red,green,blue);

    //glEnable(GL_TEXTURE_2D);
    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    //glBindTexture(GL_TEXTURE_2D, textureGL);

    runCUDA(true, true, 0);

    // load texture from PBO
    //  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
    //  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // fragment program is required to display floating point texture
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);

    glScalef(0.1,0.1,0.1);
    glRotatef(angle,0,1,0);

    glBegin(GL_TRIANGLES);

    // Iterate over each face.
    int face_index;
    for(face_index = 0; face_index < mesh.faces.size(); face_index++)
    {
    	ObjMeshFace& current_face = mesh.faces[face_index];

    	// Each face is a triangle, so draw 3 vertices with their normal
    	// and texcoords.
    	int vertex_index;
    	for(vertex_index = 0; vertex_index < 3; ++vertex_index)
    	{
    		ObjMeshVertex& vertex = current_face.vertices[vertex_index];
    		glNormal3f(vertex.normal.x, vertex.normal.y, vertex.normal.z);
    		glTexCoord2f(vertex.texcoord.x, vertex.texcoord.y);
    		glVertex3f(vertex.pos.x, vertex.pos.y, vertex.pos.z);
    	}
    }

    // End drawing of triangles.
    glEnd();



    glPopMatrix();
    angle=angle+rotation_speed;
    offset = offset + 0.01;
    if(angle>360)angle=angle-360;
    if(offset>6.28) offset = 0;
}


void display(void)
{
    glClearColor (0.0,0.0,0.0,1.0);
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    drawMesh();
    glutSwapBuffers(); //swap the buffers


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

void processNormalKeys(unsigned char key, int x, int y) {

	if (key == 27)
		exit(0);
	else if (key=='r') {

			if (rotation_speed == 0)
				rotation_speed = 0.6;
			else
				rotation_speed = 0.0;
	}
}



ObjMesh LoadObjMesh(std::string filename)
{
	ObjMesh myMesh;

	    std::vector<Vector3f>           positions;
	    std::vector<Vector2f>           texcoords;
	    std::vector<Vector3f>           normals;
	    std::vector<_ObjMeshFaceIndex>  faces;
	    /**
	     * Load file, parse it
	     * Lines beginning with:
	     * '#'  are comments can be ignored
	     * 'v'  are vertices positions (3 floats that can be positive or negative)
	     * 'vt' are vertices texcoords (2 floats that can be positive or negative)
	     * 'vn' are vertices normals   (3 floats that can be positive or negative)
	     * 'f'  are faces, 3 values that contain 3 values which are separated by / and <space>
	     */

	    std::ifstream filestream;
	    filestream.open(filename.c_str());

		std::string line_stream;
		while(std::getline(filestream, line_stream)){
			std::stringstream str_stream(line_stream);
			std::string type_str;
	        str_stream >> type_str;
	        if(type_str == TOKEN_VERTEX_POS){
	            Vector3f pos;
	            str_stream >> pos.x >> pos.y >> pos.z;
	            positions.push_back(pos);
	        }else if(type_str == TOKEN_VERTEX_TEX){
	            Vector2f tex;
	            str_stream >> tex.x >> tex.y;
	            texcoords.push_back(tex);
	        }else if(type_str == TOKEN_VERTEX_NOR){
	            Vector3f nor;
	            str_stream >> nor.x >> nor.y >> nor.z;
	            normals.push_back(nor);
	        }else if(type_str == TOKEN_FACE){
	            _ObjMeshFaceIndex face_index;
	            char interupt;
	            for(int i = 0; i < 3; ++i){
	                str_stream >> face_index.pos_index[i] >> interupt
	                           >> face_index.tex_index[i]  >> interupt
	                           >> face_index.nor_index[i];
	            }
	            faces.push_back(face_index);
	        }
	    }
		// Explicit closing of the file
	    filestream.close();

	    for(size_t i = 0; i < faces.size(); ++i){
	        ObjMeshFace face;
	        for(size_t j = 0; j < 3; ++j){
	            face.vertices[j].pos        = positions[faces[i].pos_index[j] - 1];
	            face.vertices[j].texcoord   = texcoords[faces[i].tex_index[j] - 1];
	            face.vertices[j].normal     = normals[faces[i].nor_index[j] - 1];
	        }
	        myMesh.faces.push_back(face);
	    }

	    return myMesh;
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

	BLOCKS = ceil((float)(image_width*image_height)/THREADS_PER_BLOCK);

	//cudaMalloc((int**)&d_image,image_size);
	//cudaMemcpy(d_image,image,image_size, cudaMemcpyHostToDevice);

	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(image_width,image_height);
	glutCreateWindow("Almonds Breade");

	GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f};
	GLfloat ambientLight[] = { 0.6f, 0.6f, 0.6f, 1.0f };
	GLfloat diffuseLight[] = { 1.0f, 1.0f, 1.0f, 1.0f};

	GLfloat light_position[] = { 10.0, 1.0, 1.0, 0.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT,ambientLight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,diffuseLight);
	glLightfv(GL_LIGHT0,GL_SPECULAR,specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glDisable(GL_CULL_FACE);


	// register callbacks
	//glutDisplayFunc(renderScene);
	glutDisplayFunc(display);
	glutReshapeFunc(changeSize);


	// here is the idle func registration
	//glutIdleFunc(renderScene);
	glutIdleFunc(display);

	mesh = LoadObjMesh("Elizabeth/Elizabeth.obj");

	// here are the new entries
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);
	glutSpecialUpFunc(releaseSpecialKey);


	// -----------------------------------------------------------------
	// CUDA
    // Allocate memory for renderImage (to be able to render into a CUDA memory buffer)
	cudaDeviceReset();
	cudaMalloc((void **)&d_dst, (image_width * image_height * sizeof(uchar4)));


	// enter GLUT event processing loop
	glutMainLoop();

	cudaFree(d_dst);

	return 1;
}
