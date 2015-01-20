/*
 * mesh_handler.h
 *
 *  Created on: Jan 15, 2015
 *      Author: Mike
 */

#ifndef MESH_HANDLER_H_
#define MESH_HANDLER_H_


#include <stdio.h>
#include <stdlib.h>

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


#ifdef __APPLE__

#include <GLUT/glut.h>

#else

#include <GL/freeglut.h>

#endif


typedef struct
{
    float x, y;
}Vector2f;

typedef struct
{
    float x, y, z;
} Vector3f;

typedef struct
{
    Vector3f pos;
    Vector2f texcoord;
    Vector3f normal;
} ObjMeshVertex;

/* This is a triangle, that we can render */
typedef struct
{
    ObjMeshVertex vertices[3];
} ObjMeshFace;

/* This contains a list of triangles */
typedef struct
{
    std::vector<ObjMeshFace> faces;
} ObjMesh;

/* Internal structure */
typedef struct
{
    int pos_index[3];
    int tex_index[3];
    int nor_index[3];
} _ObjMeshFaceIndex;


ObjMesh LoadObjMesh(std::string filename);

void drawMesh(ObjMesh mesh, GLuint tex_id, GLuint pbo_id, int screen_w, int screen_h, float x, float z, float r);
void drawCube(float x, float y, float z, float size, float r, float g, float b);

#endif /* MESH_HANDLER_H_ */
