/*
 * created by Michael Birkhoff
 */

#include "mesh_handler.h"


#define TOKEN_VERTEX_POS "v"
#define TOKEN_VERTEX_NOR "vn"
#define TOKEN_VERTEX_TEX "vt"
#define TOKEN_FACE "f"

void drawMesh(ObjMesh mesh, GLuint tex_id, GLuint pbo_id, int screen_w, int screen_h, float x, float z, float r)
{
    glPushMatrix();

    glTranslatef(x,-0.2f,-0.5f+z);
    glRotatef(r, 0, 1, 0);

    glScalef(0.1f,0.1f,0.1f);


    // load texture from PBO
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_id);

    glTexSubImage2D(GL_TEXTURE_2D, 0,0,0, screen_w, screen_h, GL_RGBA, GL_UNSIGNED_BYTE, 0);



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
}


void drawCube(float x, float y, float z, float size, float r, float g, float b)
{
	//glColor3f(r,g,b);
	glNormal3f(0, 0, 1);
	glVertex3f(x+size/2, y+size/2, z+size/2);

	//glColor3f(r,g,b);
	glNormal3f(0, 0, 1);
	glVertex3f(x+size/2, y-size/2, z+size/2);

	//glColor3f(r,g,b);
	glNormal3f(0, 0, 1);
	glVertex3f(x-size/2, y-size/2, z+size/2);

	//glColor3f(r,g,b);
	glNormal3f(0, 0, 1);
	glVertex3f(x-size/2, y+size/2, z+size/2);


	///

	//glColor3f(r,g,b);
	glNormal3f(1, 0, 0);
	glVertex3f(x+size/2, y+size/2, z-size/2);

	//glColor3f(r,g,b);
	glNormal3f(1, 0, 0);
	glVertex3f(x+size/2, y-size/2, z-size/2);

	//glColor3f(r,g,b);
	glNormal3f(1, 0, 0);
	glVertex3f(x+size/2, y-size/2, z+size/2);

	//glColor3f(r,g,b);
	glNormal3f(1, 0, 0);
	glVertex3f(x+size/2, y+size/2, z+size/2);

	///

	//glColor3f(r,g,b);
	glNormal3f(-1, 0, 0);
	glVertex3f(x-size/2, y+size/2, z+size/2);

	//glColor3f(r,g,b);
	glNormal3f(-1, 0, 0);
	glVertex3f(x-size/2, y-size/2, z+size/2);

	//glColor3f(r,g,b);
	glNormal3f(-1, 0, 0);
	glVertex3f(x-size/2, y-size/2, z-size/2);

	//glColor3f(r,g,b);
	glNormal3f(-1, 0, 0);
	glVertex3f(x-size/2, y+size/2, z-size/2);

	///

	//glColor3f(r,g,b);
	glNormal3f(0, 1, 0);
	glVertex3f(x-size/2, y+size/2, z+size/2);

	//glColor3f(r,g,b);
	glNormal3f(0, 1, 0);
	glVertex3f(x-size/2, y+size/2, z-size/2);

	//glColor3f(r,g,b);
	glNormal3f(0, 1, 0);
	glVertex3f(x-size/2, y+size/2, z-size/2);

	//glColor3f(r,g,b);
	glNormal3f(0, -1, 0);
	glVertex3f(x-size/2, y+size/2, z+size/2);

	//

	//glColor3f(r,g,b);
	glNormal3f(0, -1, 0);
	glVertex3f(x-size/2, y-size/2, z+size/2);

	//glColor3f(r,g,b);
	glNormal3f(0, -1, 0);
	glVertex3f(x-size/2, y-size/2, z-size/2);

	//glColor3f(r,g,b);
	glNormal3f(0, -1, 0);
	glVertex3f(x-size/2, y-size/2, z-size/2);

	//glColor3f(r,g,b);
	glNormal3f(0, -1, 0);
	glVertex3f(x-size/2, y-size/2, z+size/2);

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

	    for(size_t i = 0; i < faces.size(); ++i)
	    {
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
