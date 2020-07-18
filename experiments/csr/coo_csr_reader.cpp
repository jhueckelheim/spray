#include "coo_csr_reader.hpp"

// Parts of the MatrixMarket reder interface, declared `extern "C"`
// to work nicely with C++. Compiling mmio.c with a C++ compiler
// does not seem to work correctly.
typedef char MM_typecode[4];
extern "C" char *mm_typecode_to_str(MM_typecode matcode);
extern "C" int mm_read_banner(FILE *f, MM_typecode *matcode);
extern "C" int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
extern "C" int mm_read_mtx_array_size(FILE *f, int *M, int *N);
#define mm_is_matrix(typecode)	((typecode)[0]=='M')
#define mm_is_sparse(typecode)	((typecode)[1]=='C')
#define mm_is_complex(typecode)	((typecode)[2]=='C')

// Matrix Market file reader, slightly adapted from 
// https://math.nist.gov/MatrixMarket/mmio/c/example_read.c
// to support templated reading in single or double precision.
template <typename T>
void read_mm_coo (char* filename, coo<T>& coo_data){
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i, *I, *J;

    if ((f = fopen(filename, "r")) == NULL) {
        printf("No matrix market file name given.\n");
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) {
        printf("Error reading matrix size\n");
        exit(1);
    }


    /* reseve memory for matrices */
    coo_data.rows = (int *) malloc(nz * sizeof(int));
    coo_data.cols = (int *) malloc(nz * sizeof(int));
    coo_data.vals = (T *) malloc(nz * sizeof(T));

    coo_data.nr = N;
    coo_data.nc = M;
    coo_data.nnz = nz;

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        scanline(f, &(coo_data.rows[i]), &(coo_data.cols[i]), &(coo_data.vals[i]));
        coo_data.rows[i]--;  /* adjust from 1-based to 0-based */
        coo_data.cols[i]--;
    }

    if (f !=stdin) fclose(f);
}

void scanline(FILE *f, int *r, int *c, double *v) {
    fscanf(f, "%d %d %lg\n", r, c, v);
}

void scanline(FILE *f, int *r, int *c, float *v) {
    fscanf(f, "%d %d %g\n", r, c, v);
}

template void read_mm_coo<double> (char* filename, coo<double>& coo_data);
template void read_mm_coo<float> (char* filename, coo<float>& coo_data);

// sparsekit's coocsr conversion routine from coordinate to CSR format,
// adapted to zero-based indexing, translated to C++, and templated for
// use with different precisions or number types.
template <typename T>
void coocsr(coo<T>& coo_data, csr<T>& csr_data) {
    csr_data.rptr = (int *) malloc((coo_data.nr + 1) * sizeof(int));
    csr_data.cols = (int *) malloc(coo_data.nnz * sizeof(int));
    csr_data.vals = (T *) malloc(coo_data.nnz * sizeof(T));
    csr_data.nnz = coo_data.nnz;
    csr_data.nr = coo_data.nr;
    csr_data.nc = coo_data.nc;

    // Count entries per row.
    for (int i=0; i<csr_data.nr+1; i++) {
        csr_data.rptr[i] = 0;
    }
    for (int i=0; i<csr_data.nnz; i++) {
        csr_data.rptr[coo_data.cols[i]]++;
    }

    // Starting position for each row.
    int k = 0;
    int k0;
    for (int i=0; i<csr_data.nr+1; i++) {
        k0 = csr_data.rptr[i];
        csr_data.rptr[i] = k;
        k += k0;
    }

    // Go through the structure once more. Fill in output matrix.
    for (int i=0; i<csr_data.nnz; i++) {
        int row = coo_data.rows[i];
        int col = coo_data.cols[i];
        T val = coo_data.vals[i];
        int pos = csr_data.rptr[row];
        csr_data.vals[pos] = val;
        csr_data.cols[pos] = col;
        csr_data.rptr[row]++;
    }

    // Shift back rptr.
    for (int i=csr_data.nr; i>0; i--) {
        csr_data.rptr[i] = csr_data.rptr[i-1];
    }
    csr_data.rptr[0] = 0;
}

template void coocsr(coo<double>& coo_data, csr<double>& csr_data);
template void coocsr(coo<float>& coo_data, csr<float>& csr_data);
