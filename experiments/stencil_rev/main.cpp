#include <iostream>
#include <math.h>
#include "spray.hpp"

typedef double real;

void conv1d(int N, real* in, real* out, int S, real* wl, real* wr, real wc) {
  #pragma omp parallel for
  for(int i=S; i<N-S; i++) {
    out[i] += wc*in[i];
    for(int j=0; j<S; j++) {
      out[i] += wl[j]*in[i-j-1] + wr[j]*in[i+j+1];
    }
  }
}

void conv1d_b_serial(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb[i - j - 1] += wl[j]*outb[i];
            inb[i + j + 1] += wr[j]*outb[i];
        }
        inb[i] += wc*outb[i];
    }
}

void conv1d_b_reduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    #pragma omp parallel for reduction(+:inb[0:N])
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb[i - j - 1] += wl[j]*outb[i];
            inb[i + j + 1] += wr[j]*outb[i];
        }
        inb[i] += wc*outb[i];
    }
}

void conv1d_b_atomic(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    #pragma omp parallel for
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            #pragma omp atomic update
            inb[i - j - 1] += wl[j]*outb[i];
            #pragma omp atomic update
            inb[i + j + 1] += wr[j]*outb[i];
        }
        #pragma omp atomic update
        inb[i] += wc*outb[i];
    }
}

void conv1d_b_blockreduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::BlockReduction<real> inb_b(N,inb,false);
    #pragma omp parallel for reduction(+:inb_b)
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_lockreduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::BlockReduction<real> inb_b(N,inb,true);
    #pragma omp parallel for reduction(+:inb_b)
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_catomicreduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::AtomicReduction<real> inb_b(inb);
    #pragma omp parallel for reduction(+:inb_b)
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_cdensereduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::DenseReduction<real> inb_b(N,inb);
    #pragma omp parallel for reduction(+:inb_b)
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_keeperreduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::KeeperReduction<real> inb_b(N,inb);
    #pragma omp parallel for reduction(+:inb_b)
    for(int i=S; i<N-S; i++) {
    //for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_aw16reduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::BlockReduction16<real> inb_b(N,inb);
    #pragma omp parallel for reduction(+:inb_b)
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_aw64reduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::BlockReduction64<real> inb_b(N,inb);
    #pragma omp parallel for reduction(+:inb_b)
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_aw256reduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::BlockReduction256<real> inb_b(N,inb);
    #pragma omp parallel for reduction(+:inb_b)
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_aw1024reduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::BlockReduction1024<real> inb_b(N,inb);
    #pragma omp parallel for reduction(+:inb_b)
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_aw4096reduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::BlockReduction4096<real> inb_b(N,inb);
    #pragma omp parallel for reduction(+:inb_b)
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_aw16384reduce(int N, real *in, real *inb, real *out, real *outb, int S, real *
        wl, real *wr, real wc) {
    spray::BlockReduction16384<real> inb_b(N,inb);
    #pragma omp parallel for reduction(+:inb_b)
    for (int i = N-S-1; i > S-1; --i) {
        for (int j = S-1; j > -1; --j) {
            inb_b[i - j - 1] += wl[j]*outb[i];
            inb_b[i + j + 1] += wr[j]*outb[i];
        }
        inb_b[i] += wc*outb[i];
    }
}

void conv1d_b_repeat(int domainsize, real *domain_in, real *domain_inb, real *domain_out, real *domain_outb, int stencilsize, real *
        weightsl, real *weightsr, real weightc, int iters, int method) {
  for(int c=0; c<iters; c++) {
    switch (method)
    {
      case 1:
        conv1d_b_serial(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 2:
        conv1d_b_reduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 3:
        conv1d_b_atomic(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 4:
        conv1d_b_blockreduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 5:
        conv1d_b_lockreduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 6:
        conv1d_b_catomicreduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 7:
        conv1d_b_cdensereduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 8:
        conv1d_b_keeperreduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 9:
        conv1d_b_aw16reduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 10:
        conv1d_b_aw64reduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 11:
        conv1d_b_aw256reduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 12:
        conv1d_b_aw1024reduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 13:
        conv1d_b_aw4096reduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      case 14:
        conv1d_b_aw16384reduce(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc);
        break;
      default:
        std::cout<<"Unknown method, choose 1-14."<<std::endl;
    }
  }
}

int main(int argc, char** argv) {
  if(argc < 5) {
    std::cout<<"Usage: "<<argv[0]<<" <domainsize> <stencilsize> <method> <iters>"<<std::endl;
    return -1;
  }
  int domainsize = atoi(argv[1]);
  int stencilsize = atoi(argv[2]);
  int method = atoi(argv[3]);
  int iters = atoi(argv[4]);
  std::cout<<"Domainsize "<<domainsize<<" stencilsize "<<stencilsize<<std::endl;

  real* domain_in = (real*) malloc(domainsize * sizeof(real));
  real* domain_out = (real*) malloc(domainsize * sizeof(real));
  real* domain_inb = (real*) malloc(domainsize * sizeof(real));
  real* domain_outb = (real*) malloc(domainsize * sizeof(real));
  real* weightsl = (real*) malloc(stencilsize * sizeof(real));
  real* weightsr = (real*) malloc(stencilsize * sizeof(real));

  for(int i=0; i<domainsize; i++) {
    domain_in[i] = sin(0.1*i);
    domain_out[i] = 0.0;
  }
  for(int i=0; i<stencilsize; i++) {
    weightsl[i] = sin(0.1*i);
    weightsr[i] = cos(0.1*i);
  }
  real weightc = -2.0;

  conv1d(domainsize, domain_in, domain_out, stencilsize, weightsl, weightsr, weightc);

  real checksum = 0;
  for(int i=0; i<domainsize; i++) {
    domain_outb[i] = sin(0.1*i);
    domain_inb[i] = 0.0;
  }

  conv1d_b_repeat(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc, iters, method);
  double time = omp_get_wtime();
  conv1d_b_repeat(domainsize, domain_in, domain_inb, domain_out, domain_outb, stencilsize, weightsl, weightsr, weightc, iters, method);
  time = omp_get_wtime() - time;

  for(int i=0; i<domainsize; i++) {
    checksum += domain_inb[i];
  }
  std::cout<<"Method "<<method<<" on "<<omp_get_max_threads()<<" threads in "<<time<<" s, checksum "<<checksum<<std::endl;
  return 0;
}
