#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#define ndims 2
#define nbneighbours 4
#define le 0
#define ri 1
#define be 2
#define fr 3
#define MAX_ITERATION 1000000
#define INDEX_2D_to_1D(i, j) ((j-(yid_start-1))*(xid_end-xid_start+3) + i-(xid_start-1))
#define exactsol(x,y) x*y*(x*x+y*y)
#define Source(x,y)   12.*x*y
//#define exactsol(x,y) (x*x+y*y)
//#define Source(x,y)   4.0
#define max(X,Y) ((X) > (Y) ? (X) : (Y))
#define pi 3.1415926535897932
#define false (0==1)
#define a1 -1.
#define b1 1.
#define a2 -1.
#define b2 1.

int rank,xid_start,xid_end,yid_start,yid_end;
static int nprocs;
static MPI_Comm mycomm;
static int dims[ndims];
static int neighbour[nbneighbours];
static MPI_Datatype xslice_type, yslice_type;
static double *f;
static double coeff[3];
static double *u, *u_exact;

double absmax(double *u) 
{
  double localerror, diffnorm;
  int iterx, itery;
  localerror = 0;
  for (itery=yid_start; itery<yid_end+1; itery++) {
    for (iterx=xid_start; iterx<xid_end+1; iterx++) {
      double temp = fabs( u[INDEX_2D_to_1D(iterx, itery)] );
      if (localerror < temp) localerror = temp;
    }
  }
  MPI_Allreduce(&localerror, &diffnorm, 1, MPI_DOUBLE, MPI_MAX, mycomm);
  return diffnorm;
}

double max_error(double *u, double *u_new) 
{
  double localerror, diffnorm;
  int iterx, itery;
  localerror = 0;
  for (itery=yid_start; itery<yid_end+1; itery++) {
    for (iterx=xid_start; iterx<xid_end+1; iterx++) {
      double temp = fabs( u[INDEX_2D_to_1D(iterx, itery)] - u_new[INDEX_2D_to_1D(iterx, itery)] );
      if (localerror < temp) localerror = temp;
    }
  }
  MPI_Allreduce(&localerror, &diffnorm, 1, MPI_DOUBLE, MPI_MAX, mycomm);
  return diffnorm;
}

void jacobi2d_mpi(double *u, int*it, double eps) {
  int iterx, itery;
  if (rank ==0 & 1==2)
  printf("Welcome to the iterative Jacobi solver!\n");
  double *temp, diffnorm;
  int isconvergent;
  int niter = 0;
  isconvergent = false;
  while (!isconvergent && (niter < MAX_ITERATION)) {
    niter += 1;
    ////////////////////////////////////////////////////////////////////////////////
    const int tag1 = 0;
    MPI_Status statut;
    MPI_Sendrecv(&(u[INDEX_2D_to_1D(xid_start, yid_start)]), 1, yslice_type, neighbour[le], tag1, &(u[INDEX_2D_to_1D(xid_end+1, yid_start)]), 1, yslice_type, neighbour[ri], tag1, mycomm, &statut);
    MPI_Sendrecv(&(u[INDEX_2D_to_1D(xid_end, yid_start)]), 1, yslice_type, neighbour[ri], tag1,&(u[INDEX_2D_to_1D(xid_start-1, yid_start)]), 1, yslice_type, neighbour[le], tag1, mycomm, &statut);
    MPI_Sendrecv(&(u[INDEX_2D_to_1D(xid_start, yid_start)]), 1, xslice_type , neighbour[fr], tag1, &(u[INDEX_2D_to_1D(xid_start, yid_end+1)]),1,xslice_type,neighbour[be],tag1,mycomm,&statut);
    MPI_Sendrecv(&(u[INDEX_2D_to_1D(xid_start, yid_end)]), 1, xslice_type, neighbour[be], tag1, &(u[INDEX_2D_to_1D(xid_start, yid_start-1)]),1,xslice_type, neighbour[fr], tag1, mycomm, &statut);
    ////////////////////////////////////////////////////////////////////////////////
    double localerror = 0.0;
    for (itery=yid_start; itery<yid_end+1; itery++) {
      for (iterx=xid_start; iterx<xid_end+1; iterx++) {
        double u_new = coeff[0] * (  coeff[1]*(u[INDEX_2D_to_1D(iterx+1, itery)]+u[INDEX_2D_to_1D(iterx-1, itery)])
						      + coeff[2]*(u[INDEX_2D_to_1D(iterx, itery+1)]+u[INDEX_2D_to_1D(iterx, itery-1)]) 
						      - f[INDEX_2D_to_1D(iterx, itery)]);
        localerror = max(localerror,fabs(u_new-u[INDEX_2D_to_1D(iterx, itery)]));
        u[INDEX_2D_to_1D(iterx, itery)] = u_new;
      }
    }
    MPI_Allreduce(&localerror, &diffnorm, 1, MPI_DOUBLE, MPI_MAX, mycomm);
    isconvergent = (diffnorm < eps);
  }
  *it = niter;
}

double dot(double*u,double*v)
{
  int iterx, itery; 
  double r = 0;
  for (itery=yid_start-1; itery<yid_end+2; itery++) {
    for (iterx=xid_start-1; iterx<xid_end+2; iterx++) {
      r += u[INDEX_2D_to_1D(iterx, itery)]*v[INDEX_2D_to_1D(iterx, itery)];
    }
  }  
  return(r);
}

void cg2d_mpi(double *u, int*niter, double eps) {
  if (rank ==0 & 1==2)
  printf("Welcome to the conjugate gradient solver!\n");
  double *r0=calloc( (xid_end-xid_start+3) * (yid_end-yid_start+3),sizeof(double));
  double *p0=calloc( (xid_end-xid_start+3) * (yid_end-yid_start+3),sizeof(double));
  double *Ap=calloc( (xid_end-xid_start+3) * (yid_end-yid_start+3),sizeof(double));
  int iterx, itery;
  double rTr_prev_local = 0;

  for (itery=yid_start; itery<yid_end+1; itery++) {
    for (iterx=xid_start; iterx<xid_end+1; iterx++) {
      double Au =coeff[1]*(u[INDEX_2D_to_1D(iterx+1, itery)]-2.0*u[INDEX_2D_to_1D(iterx, itery)]+u[INDEX_2D_to_1D(iterx-1, itery)])
                 + coeff[2]*(u[INDEX_2D_to_1D(iterx, itery+1)]-2.0*u[INDEX_2D_to_1D(iterx, itery)]+u[INDEX_2D_to_1D(iterx, itery-1)]);
      r0[INDEX_2D_to_1D(iterx, itery)] = f[INDEX_2D_to_1D(iterx, itery)] - Au;
      p0[INDEX_2D_to_1D(iterx, itery)] = r0[INDEX_2D_to_1D(iterx, itery)];
      rTr_prev_local+= r0[INDEX_2D_to_1D(iterx, itery)]*r0[INDEX_2D_to_1D(iterx, itery)];
    }
  }  

  double rTr_prev=0.0;
  MPI_Allreduce(&rTr_prev_local, &rTr_prev, 1, MPI_DOUBLE, MPI_SUM, mycomm);
  int it = 0;

  int isconvergent = false;
  while (it < MAX_ITERATION) {
    MPI_Status statut;
    ////////////////////////////////////////////////////////////////////////////////
    const int tag2 = 1;
    MPI_Sendrecv(&(p0[INDEX_2D_to_1D(xid_start, yid_start)]), 1, yslice_type, neighbour[le], tag2, &(p0[INDEX_2D_to_1D(xid_end+1, yid_start)]), 1, yslice_type, neighbour[ri], tag2, mycomm, &statut);
    MPI_Sendrecv(&(p0[INDEX_2D_to_1D(xid_end, yid_start)]), 1, yslice_type, neighbour[ri], tag2,&(p0[INDEX_2D_to_1D(xid_start-1, yid_start)]), 1, yslice_type, neighbour[le], tag2, mycomm, &statut);
    MPI_Sendrecv(&(p0[INDEX_2D_to_1D(xid_start, yid_start)]), 1, xslice_type , neighbour[fr], tag2, &(p0[INDEX_2D_to_1D(xid_start, yid_end+1)]),1,xslice_type,neighbour[be],tag2,mycomm,&statut);
    MPI_Sendrecv(&(p0[INDEX_2D_to_1D(xid_start, yid_end)]), 1, xslice_type, neighbour[be], tag2, &(p0[INDEX_2D_to_1D(xid_start, yid_start-1)]),1,xslice_type, neighbour[fr], tag2, mycomm, &statut);
    ///////////////////////////////////////////////////////////////////////////////*/
    double pTAp_local = 0.0;
    for (itery=yid_start; itery<yid_end+1; itery++) {
      for (iterx=xid_start; iterx<xid_end+1; iterx++) {
        Ap[INDEX_2D_to_1D(iterx, itery)] =coeff[1]*(p0[INDEX_2D_to_1D(iterx+1, itery)]-2.0*p0[INDEX_2D_to_1D(iterx, itery)]+p0[INDEX_2D_to_1D(iterx-1, itery)])
                 + coeff[2]*(p0[INDEX_2D_to_1D(iterx, itery+1)]-2.0*p0[INDEX_2D_to_1D(iterx, itery)]+p0[INDEX_2D_to_1D(iterx, itery-1)]);
	pTAp_local += p0[INDEX_2D_to_1D(iterx, itery)]*Ap[INDEX_2D_to_1D(iterx, itery)];
      }
    }
    double pTAp; 
    MPI_Allreduce(&pTAp_local, &pTAp, 1, MPI_DOUBLE, MPI_SUM, mycomm);
    double alpha = rTr_prev/pTAp;
    double rTr_local = 0.0;
    for (itery=yid_start; itery<yid_end+1; itery++) {
      for (iterx=xid_start; iterx<xid_end+1; iterx++) {
        u[INDEX_2D_to_1D(iterx, itery)] += alpha*p0[INDEX_2D_to_1D(iterx, itery)];
        r0[INDEX_2D_to_1D(iterx, itery)]-=alpha*Ap[INDEX_2D_to_1D(iterx, itery)];
        rTr_local += r0[INDEX_2D_to_1D(iterx, itery)]*r0[INDEX_2D_to_1D(iterx, itery)] ;
      }
    } 
    double localerror = absmax(r0) ;//dot(r0,r0);
    double globalerror;
    MPI_Allreduce(&localerror, &globalerror, 1, MPI_DOUBLE, MPI_MAX, mycomm);

    isconvergent = globalerror<eps;
    if (isconvergent)
    {
      *niter = it;
      return;
    }   
    double rTr;
    MPI_Allreduce(&rTr_local, &rTr, 1, MPI_DOUBLE, MPI_SUM, mycomm);
    double beta = rTr/rTr_prev;
    rTr_prev = rTr;

    for (itery=yid_start; itery<yid_end+1; itery++) {
      for (iterx=xid_start; iterx<xid_end+1; iterx++) {
        p0[INDEX_2D_to_1D(iterx, itery)] = r0[INDEX_2D_to_1D(iterx, itery)]+ beta*p0[INDEX_2D_to_1D(iterx, itery)];
      }
    } 
    it = it+1;
  }
  *niter = it;
  free(r0);
  free(p0);
  free(Ap);
}


int main(int argc, char *argv[]) {
  char* solver = argv[1];
  int NX = atoi(argv[2]);
  int NY = atoi(argv[3]);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  ///////////////////////////////////////////////////////////////////////////////////////
  int periods[ndims];
  const int reorder=false;
  dims[0] = dims[1] = 0;
  MPI_Dims_create(nprocs, ndims, dims);
  if (rank==0&1==2)
  {
    printf("%d=%dx%d.\n",nprocs,dims[0],dims[1]);
  }
  periods[0] = periods[1] = false;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &mycomm);
  ///////////////////////////////////////////////////////////////////////////////////////
  int coords[ndims];
  MPI_Cart_coords(mycomm, rank, ndims, coords);
  xid_start = (coords[0]*NX)/dims[0]+1;
  xid_end = ((coords[0]+1)*NX)/dims[0];
  yid_start = (coords[1]*NY)/dims[1]+1;
  yid_end = ((coords[1]+1)*NY)/dims[1];
  ///////////////////////////////////////////////////////////////////////////////////////
  double dx, dy;
  int iterx, itery;
  double x, y;
  u = calloc( (xid_end-xid_start+3) * (yid_end-yid_start+3), sizeof(double));

  u_exact = calloc( (xid_end-xid_start+3) * (yid_end-yid_start+3), sizeof(double));
  f = calloc( (xid_end-xid_start+3) * (yid_end-yid_start+3), sizeof(double));
  dx = (b1-a1)/(NX+1.);
  dy = (b2-a2)/(NY+1.);
  coeff[0] = (0.5*dx*dx*dy*dy)/(dx*dx+dy*dy);
  coeff[1] = 1./(dx*dx);
  coeff[2] = 1./(dy*dy);
  for (itery=yid_start-1; itery<yid_end+2; itery++) {
    for(iterx=xid_start-1; iterx<xid_end+2; iterx++) {
      x = iterx*dx;
      y = itery*dy;
      // Compute the source
      f[INDEX_2D_to_1D(iterx, itery)] = Source(x,y);
      // Compute the exact solution
      u_exact[INDEX_2D_to_1D(iterx, itery)] = exactsol(x,y);
      // Apply the boundary conditions
      if (iterx==0 || iterx==NX+1 || itery==0 || itery==NY+1)
      {
	u[INDEX_2D_to_1D(iterx, itery)] = exactsol(x,y);
      }
    }
  }
  ///////////////////////////////////////////////////////////////////////////
  MPI_Cart_shift(mycomm, 0, 1, &(neighbour[le]), &(neighbour[ri]));
  MPI_Cart_shift(mycomm, 1, 1, &(neighbour[fr]), &(neighbour[be]));
  //printf("Process %d has four neighbours : left %d behind %d right %d front %d\n", rank, neighbour[le], neighbour[be], neighbour[ri], neighbour[fr]);
  //printf("Process %d manages indices from %d to %d in x-direction and from %d to %d in y-direction.\n",rank,xid_start,xid_end,yid_start,yid_end);
  ///////////////////////////////////////////////////////////////////////////
  MPI_Type_vector( yid_end-yid_start+1, 1, xid_end-xid_start+3, MPI_DOUBLE, &yslice_type);
  MPI_Type_commit(&yslice_type);
  MPI_Type_contiguous(xid_end-xid_start+1, MPI_DOUBLE, &xslice_type);
  MPI_Type_commit(&xslice_type);
  ///////////////////////////////////////////////////////////////////////////
  double tstart = MPI_Wtime();
  int it;
  double eps=2.e-16;
  if (solver[0]=='j' | solver[0]=='J')
  {
    jacobi2d_mpi(u,&it,eps);
  }
  else if (solver[0]=='c' | solver[0]=='C')
  {
    cg2d_mpi(u, &it, eps);
  }
  else
  {
    printf("Only Jacobi (J) and conjugate gradient (C) are available!\n");
    MPI_Abort(mycomm, 0);
  }
  double tend = MPI_Wtime();
  double error = max_error(u, u_exact);
  /*if (rank == 0) {
    if (it<MAX_ITERATION)
      printf("Converged after %d iterations.\n", it);
    else
	printf("The solver reached the maximum number of iterations.\n");
    printf("nprocs = %d, time = %f seconds, maximum error = %.1e\n",nprocs, tend-tstart,error);
    }*/
  if (rank==0)
    printf("%d %f %.1e %d\n",nprocs, tend-tstart,error,it);
  free(u);
  free(u_exact);
  free(f);
  MPI_Type_free (&yslice_type);
  MPI_Type_free (&xslice_type);
  MPI_Finalize();
  return 0;
}
