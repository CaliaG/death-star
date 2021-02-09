#pragma once

#include "ray.h"
#include "AABB.h"
#include "hit_tests.h"

struct Material;

class Hittable {
public:
	__device__ Hittable() {}
	__device__ Hittable(Material* mat) {
		material = mat;
	}
	__device__ ~Hittable() {}

	__device__ bool virtual hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;

	AABB bounding_box;
    Material* material;
};

class Triangle : public Hittable {
public:
	__device__ Triangle(vec3 _A, vec3 _B, vec3 _C, Material* mat): Hittable(mat) {
		A = _A;
		B = _B;
		C = _C;

		normal = unit_vector(cross(B-A, C-A));
		float sx = A.x() < B.x() ? A.x() : B.x();
		sx = C.x() < sx ? C.x() : sx;
		float sy = A.y() < B.y() ? A.y() : B.y();
		sy = C.y() < sy ? C.y() : sy;
		float sz = A.z() < B.z() ? A.z() : B.z();
		sz = C.z() < sz ? C.z() : sz;

		float lx = A.x() > B.x() ? A.x() : B.x();
		lx = C.x() > lx ? C.x() : lx;
		float ly = A.y() > B.y() ? A.y() : B.y();
		ly = C.y() > ly ? C.y() : ly;
		float lz = A.z() > B.z() ? A.z() : B.z();
		lz = C.z() > lz ? C.z() : lz;

		bounding_box = AABB(vec3(sx,sy,sz), vec3(lx,ly,lz));
	}

	__device__ ~Triangle() {}

	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		const float EPSILON = .0000001;
		vec3 edge1, edge2, h, s, q;
		float a, f, u, v;

		edge1 = B - A;
		edge2 = C - A;
		h = cross(r.direction, edge2);
		a = dot(edge1, h);

		if (a > -EPSILON && a < EPSILON)
			return false;  // Parallel to the triangle

		f = 1.0 / a;
		s = r.origin - A;
		u = f * dot(s, h);

		if (u < 0.0 || u > 1.0)
			return false;

		q = cross(s, edge1);
		v = f * dot(r.direction, q);

		if (v < 0.0 || u + v > 1.0)
			return false;

		float t = f * dot(edge2, q);

		if (t > t_min && t < t_max) {
			rec.t = t;
			rec.p = r.point_at_parameter(t);
			rec.normal = normal;
			rec.material = material;
			return true;
		}
		else
			return false;
	}

	vec3 A;
	vec3 B;
	vec3 C;
	vec3 normal;
};


class Sphere : public Hittable {
public:
	__device__ Sphere(vec3 center, float radius, Material* material) : Hittable(material), center(center), radius(radius) {
		bounding_box = AABB(vec3(center.x()-abs(radius), center.y()-abs(radius), center.z()-abs(radius)),
										vec3(center.x()+abs(radius), center.y()+abs(radius), center.z()+abs(radius)));

	}
	__device__ ~Sphere() {}

	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
		vec3 oc = r.origin - center;
		float a = dot(r.direction, r.direction);
		float b = dot(oc, r.direction);
		float c = dot(oc, oc) - radius*radius;
		float discriminant = b*b - a*c;

		if(discriminant > 0)
		{
			float temp = (-b - sqrt(discriminant)) / a;
			if (temp < t_max && temp > t_min)
			{
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.material = material;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp < t_max && temp > t_min)
			{
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.material = material;
				return true;
			}
		}
		return false;
	}

	vec3 center;
	float radius;
};