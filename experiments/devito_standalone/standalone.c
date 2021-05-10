#define _POSIX_C_SOURCE 200809L

#include "math.h"
#include "ndblockReduction.h"
#include "dynamicKeeperReduction.h"
#include "pmmintrin.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include <assert.h>

int Hits, Misses;

struct dataobj {
  void *restrict data;
  int *size;
  int *npsize;
  int *dsize;
  int *hsize;
  int *hofs;
  int *oofs;
};

struct profiler {
  double section0;
  double section1;
  double section2;
};

void bf0(float *damp_vec, const float dt, float *u_vec, float *vp_vec,
         const int t0, const int t1, const int t2, const int x0_blk0_size,
         const int x_M, const int x_m, const int y0_blk0_size, const int y_M,
         const int y_m, const int z_M, const int z_m);

int Forward(float *damp_vec, const float dt, const float o_x, const float o_y,
            const float o_z, float *rec_vec, float *rec_coords_vec,
            float *src_vec, int *src_gridpoints_vec,
            float *src_interpolation_coeffs_vec, float *u_vec, float *vp_vec,
            const int x_M, const int x_m, const int y_M, const int y_m,
            const int z_M, const int z_m, const int p_rec_M, const int p_rec_m,
            const int p_src_M, const int p_src_m, const int rx_M,
            const int rx_m, const int ry_M, const int ry_m, const int rz_M,
            const int rz_m, const int time_M, const int time_m,
            const int x0_blk0_size, const int y0_blk0_size) {
  /* Allocating memory for rec(4, 737)
     Allocating memory for rec_coords(737, 3)
     Allocating memory for u(3, 893, 893, 299)
     Allocating memory for src(4, 737)
     Allocating memory for src_gridpoints(737, 3)
     Allocating memory for src_interpolation_coeffs(737, 3, 64)*/
  float(*rec)[737] = (float(*)[737])rec_vec;
  float(*rec_coords)[3] = (float(*)[3])rec_coords_vec;
  float(*src)[737] = (float(*)[737])src_vec;
  int(*src_gridpoints)[3] = (int(*)[3])src_gridpoints_vec;
  float(*src_interpolation_coeffs)[3][64] =
      (float(*)[3][64])src_interpolation_coeffs_vec;
  float(*u)[893][893][299] = (float(*)[893][893][299])u_vec;
  float(*vp)[893][299] = (float(*)[893][299])vp_vec;
  float(*damp)[893][299] = (float(*)[893][299])damp_vec;
  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int time = time_m, t0 = (time) % (3), t1 = (time + 2) % (3),
           t2 = (time + 1) % (3);
       time <= time_M; time += 1, t0 = (time) % (3), t1 = (time + 2) % (3),
           t2 = (time + 1) % (3)) {
    // printf("time %i\n", time);
    /* Begin section0 */

    bf0(damp_vec, dt, u_vec, vp_vec, t0, t1, t2, x0_blk0_size,
        x_M - (x_M - x_m + 1) % (x0_blk0_size), x_m, y0_blk0_size,
        y_M - (y_M - y_m + 1) % (y0_blk0_size), y_m, z_M, z_m);
    bf0(damp_vec, dt, u_vec, vp_vec, t0, t1, t2, x0_blk0_size,
        x_M - (x_M - x_m + 1) % (x0_blk0_size), x_m,
        (y_M - y_m + 1) % (y0_blk0_size), y_M,
        y_M - (y_M - y_m + 1) % (y0_blk0_size) + 1, z_M, z_m);
    bf0(damp_vec, dt, u_vec, vp_vec, t0, t1, t2,
        (x_M - x_m + 1) % (x0_blk0_size), x_M,
        x_M - (x_M - x_m + 1) % (x0_blk0_size) + 1, y0_blk0_size,
        y_M - (y_M - y_m + 1) % (y0_blk0_size), y_m, z_M, z_m);
    bf0(damp_vec, dt, u_vec, vp_vec, t0, t1, t2,
        (x_M - x_m + 1) % (x0_blk0_size), x_M,
        x_M - (x_M - x_m + 1) % (x0_blk0_size) + 1,
        (y_M - y_m + 1) % (y0_blk0_size), y_M,
        y_M - (y_M - y_m + 1) % (y0_blk0_size) + 1, z_M, z_m);

    /* End section0 */
    /* Begin section1 */
    double timer = omp_get_wtime();
#ifdef SPRAY_NDBLOCK
    spray_ndblock_float sp_arr;
    spray_ndblock_init_float(&sp_arr, &(u[t2][0][0][0]), 893, 893, 299);
  #pragma omp parallel for reduction(+ : sp_arr)
#else
#ifdef SPRAY_KEEPER
    int NOWN = 1024;
    int* owner_array = (int*)malloc(893*sizeof(int));
    printf("owner %p\n", owner_array);
    int nthreads = omp_get_max_threads();
    int *bins = (int*) calloc(893 * nthreads , sizeof(int));
    /*memset(bins, 0, 893 * omp_get_max_threads() * sizeof(int));*/
    #pragma omp parallel default(firstprivate)
    {
      int tid = omp_get_thread_num();
      #pragma omp for
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1) {
        for (int rx = rx_m; rx <= rx_M; rx += 1) {
          int x = (int)(rx + src_gridpoints[p_src][0]) + 6;
          ++bins[x * nthreads + tid];
#if 0
          int owner =  __sync_val_compare_and_swap (&(owner_cas[x]), -1, tid);
          if(owner == -1) owner = tid;
          if(owner != lastowner) {
            ownerseqs[tid].owner[chunkcounter] = owner;
            ownerseqs[tid].owner_start[chunkcounter] = accesscounter;
            chunkcounter++;
            lastowner = owner;
          }
          accesscounter+=(ry_M-ry_m+1)*(rz_M-rz_m+1);
          if(chunkcounter > NOWN) {printf("ERROR"); exit(-1);}
#endif
        }
      }
    }
    #pragma omp parallel for default(firstprivate)
    for (int x = 0; x < 893; ++x) {
      int max = -1, maxt = -1;
      for (int t = 0; t < nthreads; ++t) {
        if (max >= bins[x * nthreads + t])
          continue;
        max = bins[x * nthreads + t];
        maxt = t;
      }
      assert(maxt >= 0 && maxt < nthreads);
      owner_array[x] = maxt;
      /*if (owner_cas[i] == -1 || owner_cas[i] == maxt || bins[i][owner_cas[i]] == max)*/
        /*continue;*/
      /*printf("x: %i has owner T%i with %i accesses but thread T%i has %i accesses\n", i, owner_cas[i], bins[i][owner_cas[i]], maxt, max);*/
    }
    /*for (int i = 0; i < 40; ++i) {*/
      /*printf("%6i : %12i : %3i ::: ", i, ownerseqs[0].owner_start[i+1] - ownerseqs[0].owner_start[i], ownerseqs[0].owner[i]);*/
      /*printf("%6i : %12i : %3i\n", i, ownerseqs[1].owner_start[i+1] -  ownerseqs[1].owner_start[i], ownerseqs[1].owner[i]);*/
    /*}*/
    timer = omp_get_wtime() - timer;
    printf("INSPECTOR TIME %lf (keeper)\n",timer);
    timer = omp_get_wtime();
    spray_keeper_float sp_arr;
    spray_keeper_init_float(&sp_arr, &(u[t2][0][0][0]), owner_array);
    #pragma omp parallel for reduction(+ : sp_arr)
#else
  #pragma omp parallel for
#endif
#endif
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1) {
      for (int rx = rx_m; rx <= rx_M; rx += 1) {
        for (int ry = ry_m; ry <= ry_M; ry += 1) {
          for (int rz = rz_m; rz <= rz_M; rz += 1) {
            int x = (int)(rx + src_gridpoints[p_src][0]) + 6;
            int y = (int)(ry + src_gridpoints[p_src][1]) + 6;
            int z = (int)(rz + src_gridpoints[p_src][2]) + 6;
            float mag = (dt * dt) * (vp[x][y][z] * vp[x][y][z]) *
                        src[time][p_src] *
                        src_interpolation_coeffs[p_src][0][rx] *
                        src_interpolation_coeffs[p_src][1][ry] *
                        src_interpolation_coeffs[p_src][2][rz];
            #ifdef SPRAY_NDBLOCK
              spray_ndblock_increment_float(&sp_arr, x, y, z, mag);
            #else
	    #ifdef SPRAY_KEEPER
              spray_keeper_increment_float(&sp_arr, x, x*893*299+y*299+z, mag);
	    #else
              #pragma omp atomic update
              u[t2][x][y][z] += mag;
            #endif
            #endif
          }
        }
      }
    }
#ifdef SPRAY_KEEPER
    spray_keeper_finalize_float(&sp_arr);
#endif
    timer = omp_get_wtime() - timer;
#ifdef SPRAY_NDBLOCKS
    #ifdef USELOCKS
        printf("SRC TIME %lf (spray_uselocks bsize %d)\n",timer, BSIZE);
    #else
        printf("SRC TIME %lf (spray bsize %d)\n",timer, BSIZE);
    #endif
#else
#ifdef SPRAY_KEEPER
    printf("SRC TIME %lf (keeper)\n",timer);
#else
    printf("SRC TIME %lf (orig)\n",timer);
#endif
#endif

    /* End section1 */
    /* Begin section2 */

    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1) {
      float posx = -o_x + rec_coords[p_rec][0];
      float posy = -o_y + rec_coords[p_rec][1];
      float posz = -o_z + rec_coords[p_rec][2];
      int ii_rec_0 = (int)(floor(4.0e-2 * posx));
      int ii_rec_1 = (int)(floor(4.0e-2 * posy));
      int ii_rec_2 = (int)(floor(4.0e-2 * posz));
      int ii_rec_3 = (int)(floor(4.0e-2 * posz)) + 1;
      int ii_rec_4 = (int)(floor(4.0e-2 * posy)) + 1;
      int ii_rec_5 = (int)(floor(4.0e-2 * posx)) + 1;
      float px = (float)(posx - 2.5e+1F * (int)(floor(4.0e-2F * posx)));
      float py = (float)(posy - 2.5e+1F * (int)(floor(4.0e-2F * posy)));
      float pz = (float)(posz - 2.5e+1F * (int)(floor(4.0e-2F * posz)));
      float sum = 0.0F;
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 &&
          ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1) {
        sum += (-6.4e-5F * px * py * pz + 1.6e-3F * px * py +
                1.6e-3F * px * pz - 4.0e-2F * px + 1.6e-3F * py * pz -
                4.0e-2F * py - 4.0e-2F * pz + 1) *
               u[t0][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_2 + 6];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 &&
          ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1) {
        sum += (6.4e-5F * px * py * pz - 1.6e-3F * px * pz - 1.6e-3F * py * pz +
                4.0e-2F * pz) *
               u[t0][ii_rec_0 + 6][ii_rec_1 + 6][ii_rec_3 + 6];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 &&
          ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1) {
        sum += (6.4e-5F * px * py * pz - 1.6e-3F * px * py - 1.6e-3F * py * pz +
                4.0e-2F * py) *
               u[t0][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_2 + 6];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 &&
          ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1) {
        sum += (-6.4e-5F * px * py * pz + 1.6e-3F * py * pz) *
               u[t0][ii_rec_0 + 6][ii_rec_4 + 6][ii_rec_3 + 6];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 &&
          ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1) {
        sum += (6.4e-5F * px * py * pz - 1.6e-3F * px * py - 1.6e-3F * px * pz +
                4.0e-2F * px) *
               u[t0][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_2 + 6];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 &&
          ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1) {
        sum += (-6.4e-5F * px * py * pz + 1.6e-3F * px * pz) *
               u[t0][ii_rec_5 + 6][ii_rec_1 + 6][ii_rec_3 + 6];
      }
      if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 &&
          ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1) {
        sum += (-6.4e-5F * px * py * pz + 1.6e-3F * px * py) *
               u[t0][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_2 + 6];
      }
      if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 &&
          ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1) {
        sum += 6.4e-5F * px * py * pz *
               u[t0][ii_rec_5 + 6][ii_rec_4 + 6][ii_rec_3 + 6];
      }
      rec[time][p_rec] = sum;
    }

    /* End section2 */
  }

  return 0;
}

void bf0(float *damp_vec, const float dt, float *u_vec, float *vp_vec,
         const int t0, const int t1, const int t2, const int x0_blk0_size,
         const int x_M, const int x_m, const int y0_blk0_size, const int y_M,
         const int y_m, const int z_M, const int z_m) {
  float(*u)[893][893][299] = (float(*)[893][893][299])u_vec;
  float(*vp)[893][299] = (float(*)[893][299])vp_vec;
  float(*damp)[893][299] = (float(*)[893][299])damp_vec;
  #pragma omp parallel
  {
  #pragma omp for collapse(2) schedule(dynamic,1)
  for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size) {
    for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size) {
      for (int x = x0_blk0; x <= x0_blk0 + x0_blk0_size - 1; x += 1) {
        for (int y = y0_blk0; y <= y0_blk0 + y0_blk0_size - 1; y += 1) {
          #pragma omp simd aligned(damp, u, vp : 16)
          for (int z = z_m; z <= z_M; z += 1) {
            float r2 = 1.0F / dt;
            float r1 = 1.0F / (dt * dt);
            float r0 =
                1.0F / (vp[x + 6][y + 6][z + 6] * vp[x + 6][y + 6][z + 6]);
            u[t2][x + 6][y + 6][z + 6] =
                (r0 * (-r1 * (-2.0F * u[t0][x + 6][y + 6][z + 6] +
                              u[t1][x + 6][y + 6][z + 6])) +
                 r2 * (damp[x + 1][y + 1][z + 1] * u[t0][x + 6][y + 6][z + 6]) +
                 1.77777773e-5F *
                     (u[t0][x + 3][y + 6][z + 6] + u[t0][x + 6][y + 3][z + 6] +
                      u[t0][x + 6][y + 6][z + 3] + u[t0][x + 6][y + 6][z + 9] +
                      u[t0][x + 6][y + 9][z + 6] + u[t0][x + 9][y + 6][z + 6]) -
                 2.39999994e-4F *
                     (u[t0][x + 4][y + 6][z + 6] + u[t0][x + 6][y + 4][z + 6] +
                      u[t0][x + 6][y + 6][z + 4] + u[t0][x + 6][y + 6][z + 8] +
                      u[t0][x + 6][y + 8][z + 6] + u[t0][x + 8][y + 6][z + 6]) +
                 2.39999994e-3F *
                     (u[t0][x + 5][y + 6][z + 6] + u[t0][x + 6][y + 5][z + 6] +
                      u[t0][x + 6][y + 6][z + 5] + u[t0][x + 6][y + 6][z + 7] +
                      u[t0][x + 6][y + 7][z + 6] + u[t0][x + 7][y + 6][z + 6]) -
                 1.30666663e-2F * u[t0][x + 6][y + 6][z + 6]) /
                (r0 * r1 + r2 * damp[x + 1][y + 1][z + 1]);
          }
        }
      }
    }
  }
  }
}

void readfile(void *ptr, const char *name, const char *flags, size_t size,
              size_t nmemb) {
  memset(ptr, 0, size * nmemb);
  FILE *fn = fopen(name, flags);
  size_t read_nmemb = fread(ptr, size, nmemb, fn);
  fclose(fn);
  if (read_nmemb != nmemb) {
    printf("Read only %lu instead of %lu items from %s\n", read_nmemb, nmemb,
           name);
  }
}

float norm(float *array, long length) {
    double sum = 0.0;

    for(long i=0; i<length; i++) {
        sum += array[i]*array[i];
    }

    return (float) sqrt(sum);
}

int main() {
  /* Allocating memory for rec(4, 737)
Allocating memory for rec_coords(737, 3)
Allocating memory for u(3, 893, 893, 299)
Allocating memory for src(4, 737)
Allocating memory for src_gridpoints(737, 3)
Allocating memory for src_interpolation_coeffs(737, 3, 64)*/

  float *rec = malloc(sizeof(float) * 4 * 737);
  float(*rec_coords) = malloc(sizeof(float) * 3 * 737);
  float(*src) = malloc(sizeof(float) * 4 * 737);
  int(*src_gridpoints) = malloc(sizeof(int) * 3 * 737);
  float(*src_interpolation_coeffs) = malloc(sizeof(float) * 737 * 3 * 64);
  float(*u) = malloc(sizeof(float) * 3 * 893 * 893 * 299);
  float(*vp) = malloc(sizeof(float) * 893 * 893 * 299);
  float(*damp) = malloc(sizeof(float) * 893 * 893 * 299);

  readfile(rec, "rec.bin", "rb", sizeof(*rec), 4 * 737);

  readfile(rec_coords, "rec_coords.bin", "rb", sizeof(*rec_coords), 3 * 737);

  readfile(src, "src.bin", "rb", sizeof(*src), 4 * 737);

  readfile(damp, "damp.bin", "rb", sizeof(*damp), 883 * 883 * 289);

  readfile(src_gridpoints, "src_gridpoints.bin", "rb", sizeof(*src_gridpoints),
           737 * 3);

  readfile(src_interpolation_coeffs, "src_interpolation_coeffs.bin", "rb",
           sizeof(*src_interpolation_coeffs), 737 * 3 * 64);

  readfile(u, "u.bin", "rb", sizeof(*u), 3 * 893 * 893 * 299);

  readfile(vp, "vp.bin", "rb", sizeof(*vp), 893 * 893 * 299);

  float dt = 1.75;
  float o_x = -1000.0;
  float o_y = -1000.0;
  float o_z = -1000.0;
  int x_M = 880;
  int x_m = 0;
  int y_M = 880;
  int y_m = 0;
  int z_M = 286;
  int z_m = 0;
  int p_rec_M = 736;
  int p_rec_m = 0;
  int p_src_M = 736;
  int p_src_m = 0;
  int rx_M = 63;
  int rx_m = 0;
  int ry_M = 63;
  int ry_m = 0;
  int rz_M = 63;
  int rz_m = 0;
  int time_M = 2;
  int time_m = 1;
  int x0_blk0_size = 8;
  int y0_blk0_size = 8;
  Forward(damp, dt, o_x, o_y, o_z, rec, rec_coords, src, src_gridpoints,
          src_interpolation_coeffs, u, vp, x_M, x_m, y_M, y_m, z_M, z_m,
          p_rec_M, p_rec_m, p_src_M, p_src_m, rx_M, rx_m, ry_M, ry_m, rz_M,
          rz_m, time_M, time_m, x0_blk0_size, y0_blk0_size);
  float l2 = norm(u, 3 * 893 * 893 * 299);
  printf("Norm of u: %f\n", l2);
  free(rec);
  free(rec_coords);
  free(src);
  free(src_gridpoints);
  free(src_interpolation_coeffs);
  free(u);
  free(vp);
  free(damp);
  printf("Hits %i, Misses %i\n", Hits, Misses);
  return 0;
}
