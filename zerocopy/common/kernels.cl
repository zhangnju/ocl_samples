/*
Copyright (c) 2014, Syoyo Fujita

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or 
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without 
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/


#define M_PI 3.14159265


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// OCL 1.2 related code
/////////////////////////////////////////////////////////////////////////////////////////////////////////

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
	vec3 p;vec3 n;
} plane;

typedef struct _ray
{
	vec3 orig; 
	vec3 dir;
} ray;

float vdot(vec3 *v0, vec3 *v1)
{
	return (v0->x * v1->x + v0->y * v1->y + v0->z * v1->z);
}

void vcross(vec3 *c, vec3 *v0, vec3 *v1)
{
	c->x = v0->y * v1->z - v0->z * v1->y;
	c->y = v0->z * v1->x - v0->x * v1->z;
	c->z = v0->x * v1->y - v0->y * v1->x;

}

void vnormalize(vec3 *v)
{
	float len = vdot(v, v);
	len = sqrt(len);

	if(len > 1.0e-17f)
	{
		v->x /= len;
		v->y /= len;
		v->z /= len;
	}
}


void ray_sphere_intersect(intersect_pt *isect, ray *ray, sphere *sphere)
{
	vec3 rs;
	float B,C,D;

	rs.x = ray->orig.x - sphere->center.x;
	rs.y = ray->orig.y - sphere->center.y;
	rs.z = ray->orig.z - sphere->center.z;

	B = vdot(&rs, &(ray->dir));
	C = vdot(&rs, &rs) - sphere->radius * sphere->radius;
	D = B * B - C;

	if(D > 0.0f)
	{
		float t= -B - sqrt(D);

		if((t > 0.0f) && (t < isect->t ))
		{
			isect->t = t;
			isect->hit = 1;

			isect->p.x = ray->orig.x + ray->dir.x * t;
			isect->p.y = ray->orig.y + ray->dir.y * t;
			isect->p.z = ray->orig.z + ray->dir.z * t;
		
			isect->n.x = isect->p.x - sphere->center.x;
			isect->n.y = isect->p.y - sphere->center.y;
			isect->n.z = isect->p.z - sphere->center.z;

			vnormalize(&(isect->n));
		}
	}
}


void ray_plane_intersect(intersect_pt *isect, ray *ray, plane *plane)
{
	float d, v, t;
	d = -vdot(&(plane->p), &(plane->n));
	v = vdot(&ray->dir, &(plane->n));

	if(fabs(v) < 1.0e-17f) return;

	t = -(vdot(&(ray->orig), &(plane->n)) + d) / v;

	if((t > 0.0f) && (t < isect->t))
	{
		isect->t = t;
		isect->hit = 1;

		isect->p.x = ray->orig.x + ray->dir.x * t;
		isect->p.y = ray->orig.y + ray->dir.y * t;
		isect->p.z = ray->orig.z + ray->dir.z * t;

		isect->n.x = plane->n.x;
		isect->n.y = plane->n.y;
		isect->n.z = plane->n.z;
	}
}

void orthoBasis(vec3 *basis, vec3 *n)
{
	basis[2].x = n->x;
	basis[2].y = n->y;
	basis[2].z = n->z;
	basis[1].x = 0.0f;
	basis[1].y = 0.0f;
	basis[1].z = 0.0f;

	if((n->x < 0.6f) && (n->x > -0.6f))
	{
		basis[1].x = 1.0f;
	}
	else if((n->y < 0.6f) && (n->y > -0.6f))
	{
		basis[1].y = 1.0f;
	}
	else if((n->z < 0.6f) && (n->z > -0.6f))
	{
		basis[1].z = 1.0f;
	}
	else
	{
		basis[1].x = 1.0f;
	}

	vcross(&basis[0], &basis[1], &basis[2]);
	vnormalize(&basis[0]);

	vcross(&basis[1], &basis[2], &basis[0]);
	vnormalize(&basis[1]);


}

//intel OCL fmod used to be slow, verify still needed
float myfmod(float x, float y)
{
	return x - y*trunc(x/y);
}

#define NAO_SAMPLES 8;
//#define NSUBSAMPLES 1;


void ambient_occlusion(constant sphere *spheres, plane *planes, vec3 *col, intersect_pt *isect, int *seed)
{
	int i, j;
	int ntheta = NAO_SAMPLES;
	int nphi = NAO_SAMPLES;
	float eps = .0001f;

	vec3 p;
	vec3 basis[3];
	float occlusion = 0.0f;

	if(get_local_id(0)==0)
	{

	/*	if(get_local_id(1)==0)
		{
			printf("id.x = %d, id.y=%d\n" ,x,y); 
			printf("spheres: %.2f %.2f %.2f\n", spheres[0].center.x, spheres[0].center.y, spheres[0].center.z);
			printf("spheres: %.2f %.2f %.2f\n", spheres[1].center.x, spheres[1].center.y, spheres[1].center.z);
			printf("spheres: %.2f %.2f %.2f\n", spheres[2].center.x, spheres[2].center.y, spheres[2].center.z);
			printf("planes = %.2f %.2f %.2f\n", planes->p.x, planes->p.y, planes->p.z);
			printf("h=%d w=%d numsubsamples=%d\n", h, w, nsubsamples);
		}
		*/
	}
	p.x = isect->p.x + eps * isect->n.x;
	p.y = isect->p.y + eps * isect->n.y;
	p.z = isect->p.z + eps * isect->n.z;

	orthoBasis(basis, &(isect->n));

	for(j=0;j<ntheta;j++)
	{
		for(i=0;i<nphi;i++)
		{
			*seed = (int)(fmod((float)(*seed)*1364.0f+626.0f, 509.0f));
			float rand1 = (*seed)/(509.0f);
			
			*seed = (int)(fmod((float)(*seed)*1364.0f+626.0f, 509.0f));
			float rand2 = (*seed)/(509.0f);

			float theta = sqrt(rand1);
			float phi = 2.0f * M_PI * rand2;

			float x = cos(phi) * theta;
			float y = sin(phi) * theta;
			float z = sqrt(1.0f - theta * theta);

			//local->global
			float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
			float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
			float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

			ray ray1;
			intersect_pt occ_intersect;

			ray1.orig = p;
			ray1.dir.x = rx;
			ray1.dir.y = ry;
			ray1.dir.z = rz;

			occ_intersect.t = 1.0e+17f;
			occ_intersect.hit = 0;

			sphere sphere0 = spheres[0];
			sphere sphere1 = spheres[1];
			sphere sphere2 = spheres[2];
			plane *plane0 = planes;

			ray_sphere_intersect(&occ_intersect, &ray1, &sphere0);
			ray_sphere_intersect(&occ_intersect, &ray1, &sphere1);
			ray_sphere_intersect(&occ_intersect, &ray1, &sphere2);
			ray_plane_intersect(&occ_intersect, &ray1, plane0);
			
			if(occ_intersect.hit) 
			{
				//printf("ambient occlusion hit!\n");
				occlusion += 1.0f;
			}

			}
		}

		occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);

		col->x = occlusion;
		col->y = occlusion;
		col->z = occlusion;

}



//kernel void traceOnePixel(global *fimg, constant sphere *sph, constant plane *plane, int w, int h, int nSubSamples)
kernel void traceOnePixel(global float *fimg, constant sphere *spheres, constant plane *planes, int h, int w, int nsubsamples)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	float temp = (x+ get_global_size(0)) * (y+get_global_size(1)) * 4525434.0f ;
	int seed = (int)(fmod(temp, 65536.0f));

	int u,v;
 
	if(get_global_id(0) == 0)
	{
		if(get_global_id(1) == 0)
		{
			printf("w=%d h=%d numsubsamples=%d\n", w, h, nsubsamples);
		}
	}

	for(v = 0; v <  nsubsamples; v++)
	{
		for(u = 0;u< nsubsamples; u++)
		{
			float px = (x + (u/(float)nsubsamples) - (w/2.0f)) / (w/2.0f);
			float py = -(y + (v/(float)nsubsamples) - (h/2.0f)) / (h/2.0f);

			ray ray1;
			intersect_pt isect;

			ray1.orig.x = 0.0f;
			ray1.orig.y = 0.0f;
			ray1.orig.z = 0.0f;

			ray1.dir.x = px;
			ray1.dir.y = py;
			ray1.dir.z = -1.0f;
			vnormalize(&ray1.dir);

			isect.t = 1.0e+17f;
			isect.hit = 0;

			sphere sphere0 = spheres[0];
			sphere sphere1 = spheres[1];
			sphere sphere2 = spheres[2];
			plane plane0;
			plane0.p.x = planes->p.x;
			plane0.p.y = planes->p.y;
			plane0.p.z = planes->p.z;
			plane0.n.x = planes->n.x;
			plane0.n.y = planes->n.y;
			plane0.n.z = planes->n.z;

			ray_sphere_intersect(&isect, &ray1, &sphere0);
			ray_sphere_intersect(&isect, &ray1, &sphere1);
			ray_sphere_intersect(&isect, &ray1, &sphere2);
			ray_plane_intersect(&isect, &ray1, &plane0);

			if(isect.hit)
			{
				vec3 col;
				ambient_occlusion(spheres, &plane0, &col, &isect, &seed);
				fimg[3 * (y * w + x) + 0] += col.x;
				fimg[3 * (y * w + x) + 1] += col.y;
				fimg[3 * (y * w + x) + 2] += col.z;

			}

		}
	}

	fimg[3 * (y * w + x) + 0] /= (float)(nsubsamples * nsubsamples);
	fimg[3 * (y * w + x) + 1] /= (float)(nsubsamples * nsubsamples);
	fimg[3 * (y * w + x) + 2] /= (float)(nsubsamples * nsubsamples);

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// STUFF FOR LATER
/////////////////////////////////////////////////////////////////////////////////////////////////////////



__kernel void templateKernel(__global unsigned int *output, 
							__global unsigned int *input,
							const unsigned int multiplier)
{

	uint tid = get_global_id(0);

	output[tid] = input[tid] * multiplier;

}