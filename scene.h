#pragma once

struct Hittable;

struct Scene {
	int num_hittables;
	Hittable* hittables;
};
