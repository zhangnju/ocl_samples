
#ifndef SCENE_H
#define SCENE_H

//scene structures, must stay same as in AOBench.cl
//TODO: should just share the same file right?, then just inclucde in each via #include

typedef struct _vec3
{
	float x;
	float y;
	float z;
} vec3;

typedef struct _intersect_pt
{
	float t;
	vec3 p;
	vec3 n;
	int hit;
} intersect_pt;

typedef struct _sphere
{
	vec3 center;
	float radius;
} sphere;


typedef struct _plane
{
	vec3 p;
	vec3 n;
} plane;

typedef struct _ray
{
	vec3 orig; 
	vec3 dir;
} ray;


//THese are not used host side, placeholders for now, remove before shipping if not needed
//NEED TO GO TO A SCENE.CPP FILE IF NEEDED
/*
float vdot()
{
	return 1.0f;
}

float vcross()
{
	return 1.0f;
}

float vnormalize()
{
	return 1.0f;
}
*/
//end scene structs



#endif //_SCENE_H_