#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#define ndims 3
#define nbneighbours 6
#define le 0
#define ri 1
#define be 2
#define fr 3
#define bo 4
#define to 5
#define MAX_ITERATION 1000000
#define INDEX_3D_to_1D(i,j,k) ( (k-(zid_start-1))*(yid_end-yid_start+3)*(xid_end-xid_start+3)+(j-(yid_start-1))*(xid_end-xid_start+3) + i-(xid_start-1))
#define exactsol(x,y,z) x*y*z*(x-1.)*(y-1.)*(z-1.)
#define Source(x,y,z)   2.*y*z*(y-1.)*(z-1.)+2.*x*z*(x-1.)*(z-1.)+2.*x*y*(x-1.)*(y-1.)
#define max(X,Y) ((X) > (Y) ? (X) : (Y))
#define pi 3.1415926535897932
#define false (0==1)
#define a1 -1.
#define b1 2.
#define a2 -2.
#define b2 2.
#define a3 -3.
#define b3 2.

int rank,xid_start,xid_end,yid_start,yid_end,zid_start,zid_end;
static int nprocs;
static MPI_Comm mycomm;
static int dims[ndims];
static int neighbour[nbneighbours];
static MPI_Datatype xyslice_type, yzslice_type, zxslice_type;
static double *f;
static double coeff[4];
double dx, dy, dz;

double absmax(double *u) 
{
  double localerror, diffnorm;
  int iterx, itery, iterz;
  localerror = 0;
  for (iterz=zid_start; iterz<zid_end+1; iterz++) {
    for (itery=yid_start; itery<yid_end+1; itery++) {
      for (iterx=xid_start; iterx<xid_end+1; iterx++) {
        double temp = fabs( u[INDEX_3D_to_1D(iterx,itery,iterz)] );
        if (localerror < temp) localerror = temp;
      }
    }
  }
  MPI_Allreduce(&localerror, &diffnorm, 1, MPI_DOUBLE, MPI_MAX, mycomm);
  return diffnorm;
}

double max_error(double *u, double *u_new) 
{
  double localerror, diffnorm;
  int iterx, itery, iterz;
  localerror = 0;
  for (iterz=zid_start; iterz<zid_end+1; iterz++) {
    for (itery=yid_start; itery<yid_end+1; itery++) {
      for (iterx=xid_start; iterx<xid_end+1; iterx++) {
        double temp = fabs( u[INDEX_3D_to_1D(iterx, itery,iterz)] - u_new[INDEX_3D_to_1D(iterx, itery,iterz)] );
        if (localerror < temp) localerror = temp;
      }
    }
  }
  MPI_Allreduce(&localerror, &diffnorm, 1, MPI_DOUBLE, MPI_MAX, mycomm);
  return diffnorm;
}

void jacobi3d_mpi(double *u, int*niter, double eps) {
  int iterx, itery, iterz;
  int it = 0;
  double diffnorm;
  int isconvergent = false;
  while (!isconvergent && (it < MAX_ITERATION)) {
    it = it+1;
    ////////////////////////////////////////////////////////////////////////////////
    const int tag = 0;
    MPI_Status statut;

    MPI_Sendrecv(&(u[INDEX_3D_to_1D(xid_start, yid_start,zid_start)]), 1, yzslice_type, neighbour[le], tag, &(u[INDEX_3D_to_1D(xid_end+1, yid_start,zid_start)]), 1, yzslice_type, neighbour[ri], tag, mycomm, &statut);
    MPI_Sendrecv(&(u[INDEX_3D_to_1D(xid_end,yid_start,zid_start)]), 1, yzslice_type, neighbour[ri], tag,&(u[INDEX_3D_to_1D(xid_start-1, yid_start,zid_start)]), 1, yzslice_type, neighbour[le], tag, mycomm, &statut);

    MPI_Sendrecv(&(u[INDEX_3D_to_1D(xid_start, yid_start,zid_start)]), 1, zxslice_type , neighbour[fr], tag, &(u[INDEX_3D_to_1D(xid_start, yid_end+1,zid_start)]),1,zxslice_type,neighbour[be],tag,mycomm,&statut);
    MPI_Sendrecv(&(u[INDEX_3D_to_1D(xid_start, yid_end, zid_start)]), 1, zxslice_type, neighbour[be], tag, &(u[INDEX_3D_to_1D(xid_start, yid_start-1,zid_start)]),1,zxslice_type, neighbour[fr], tag, mycomm, &statut);

    MPI_Sendrecv(&(u[INDEX_3D_to_1D(xid_start, yid_start,zid_start)]), 1, xyslice_type , neighbour[bo], tag, &(u[INDEX_3D_to_1D(xid_start, yid_start,zid_end+1)]),1,xyslice_type,neighbour[to],tag,mycomm,&statut);
    MPI_Sendrecv(&(u[INDEX_3D_to_1D(xid_start, yid_start, zid_end)]), 1, xyslice_type, neighbour[to], tag, &(u[INDEX_3D_to_1D(xid_start, yid_start,zid_start-1)]),1,xyslice_type, neighbour[bo], tag, mycomm, &statut);///
    ////////////////////////////////////////////////////////////////////////////////
    double local_max=0.0;
    for (iterz=zid_start; iterz<zid_end+1; iterz++) {
      for (itery=yid_start; itery<yid_end+1; itery++) {
        for(iterx=xid_start; iterx<xid_end+1; iterx++) {
        	  double u_new = coeff[0] * (  coeff[1]*(u[INDEX_3D_to_1D(iterx+1,itery,iterz)]+u[INDEX_3D_to_1D(iterx-1, itery,iterz)]) 
							          + coeff[2]*(u[INDEX_3D_to_1D(iterx, itery+1,iterz)]+u[INDEX_3D_to_1D(iterx, itery-1,iterz)])
      							          + coeff[3]*(u[INDEX_3D_to_1D(iterx, itery,iterz+1)]+u[INDEX_3D_to_1D(iterx, itery,iterz-1)])
							          - f[INDEX_3D_to_1D(iterx, itery,iterz)] );
            local_max = max(local_max,fabs(u_new-u[INDEX_3D_to_1D(iterx, itery,iterz)]));
            u[INDEX_3D_to_1D(iterx, itery,iterz)] = u_new;
	  }
      }
    }
    MPI_Allreduce(&local_max, &diffnorm, 1, MPI_DOUBLE, MPI_MAX, mycomm);
    isconvergent = (diffnorm < eps);
  }
  *niter = it;
}

void cg3d_mpi(double *u, int*niter, double eps) {
  if (rank ==0 & 1==2)
  printf("Welcome to the conjugate gradient solver!\n");
  double *r0=calloc( (xid_end-xid_start+3)*(yid_end-yid_start+3)*(zid_end-zid_start+3),sizeof(double));
  double *p0=calloc( (xid_end-xid_start+3)*(yid_end-yid_start+3)*(zid_end-zid_start+3),sizeof(double));
  double *Ap=calloc( (xid_end-xid_start+3)*(yid_end-yid_start+3)*(zid_end-zid_start+3),sizeof(double));
  int iterx,itery,iterz;
  double rTr_prev_local = 0;

  for (iterz=zid_start; iterz<zid_end+1; iterz++) {
    for (itery=yid_start; itery<yid_end+1; itery++) {
      for (iterx=xid_start; iterx<xid_end+1; iterx++) {
        double Au =  coeff[1]*(u[INDEX_3D_to_1D(iterx+1,itery,iterz)]-2.0*u[INDEX_3D_to_1D(iterx, itery,iterz)]+u[INDEX_3D_to_1D(iterx-1, itery,iterz)])
                 + coeff[2]*(u[INDEX_3D_to_1D(iterx,itery+1,iterz)]-2.0*u[INDEX_3D_to_1D(iterx, itery,iterz)]+u[INDEX_3D_to_1D(iterx, itery-1,iterz)])
                 + coeff[3]*(u[INDEX_3D_to_1D(iterx,itery,iterz+1)]-2.0*u[INDEX_3D_to_1D(iterx, itery,iterz)]+u[INDEX_3D_to_1D(iterx, itery,iterz-1)]);
        r0[INDEX_3D_to_1D(iterx, itery,iterz)] = f[INDEX_3D_to_1D(iterx, itery,iterz)] - Au;
        p0[INDEX_3D_to_1D(iterx, itery,iterz)] = r0[INDEX_3D_to_1D(iterx, itery,iterz)];
        rTr_prev_local+= r0[INDEX_3D_to_1D(iterx,itery,iterz)]*r0[INDEX_3D_to_1D(iterx,itery,iterz)];
      }
    }
  }  

  double rTr_prev=0.0;
  MPI_Allreduce(&rTr_prev_local, &rTr_prev, 1, MPI_DOUBLE, MPI_SUM, mycomm);
  int it = 0;

  int isconvergent = false;
  while (it < MAX_ITERATION) {
    ////////////////////////////////////////////////////////////////////////////////
    const int tag = 0;
    MPI_Status statut;
    MPI_Sendrecv(&(p0[INDEX_3D_to_1D(xid_start, yid_start,zid_start)]), 1, yzslice_type, neighbour[le], tag, &(p0[INDEX_3D_to_1D(xid_end+1, yid_start,zid_start)]), 1, yzslice_type, neighbour[ri], tag, mycomm, &statut);
    MPI_Sendrecv(&(p0[INDEX_3D_to_1D(xid_end,yid_start,zid_start)]), 1, yzslice_type, neighbour[ri], tag,&(p0[INDEX_3D_to_1D(xid_start-1, yid_start,zid_start)]), 1, yzslice_type, neighbour[le], tag, mycomm, &statut);

    MPI_Sendrecv(&(p0[INDEX_3D_to_1D(xid_start, yid_start,zid_start)]), 1, zxslice_type , neighbour[fr], tag, &(p0[INDEX_3D_to_1D(xid_start, yid_end+1,zid_start)]),1,zxslice_type,neighbour[be],tag,mycomm,&statut);
    MPI_Sendrecv(&(p0[INDEX_3D_to_1D(xid_start, yid_end, zid_start)]), 1, zxslice_type, neighbour[be], tag, &(p0[INDEX_3D_to_1D(xid_start, yid_start-1,zid_start)]),1,zxslice_type, neighbour[fr], tag, mycomm, &statut);

    MPI_Sendrecv(&(p0[INDEX_3D_to_1D(xid_start, yid_start,zid_start)]), 1, xyslice_type , neighbour[bo], tag, &(p0[INDEX_3D_to_1D(xid_start, yid_start,zid_end+1)]),1,xyslice_type,neighbour[to],tag,mycomm,&statut);
    MPI_Sendrecv(&(p0[INDEX_3D_to_1D(xid_start, yid_start, zid_end)]), 1, xyslice_type, neighbour[to], tag, &(p0[INDEX_3D_to_1D(xid_start, yid_start,zid_start-1)]),1,xyslice_type, neighbour[bo], tag, mycomm, &statut);///
    ////////////////////////////////////////////////////////////////////////////////
    double pTAp_local = 0.0;
    for (iterz=zid_start; iterz<zid_end+1; iterz++) {
      for (itery=yid_start; itery<yid_end+1; itery++) {
        for (iterx=xid_start; iterx<xid_end+1; iterx++) {
          Ap[INDEX_3D_to_1D(iterx, itery,iterz)] =coeff[1]*(p0[INDEX_3D_to_1D(iterx+1, itery,iterz)]-2.0*p0[INDEX_3D_to_1D(iterx, itery,iterz)]+p0[INDEX_3D_to_1D(iterx-1, itery,iterz)])
                 + coeff[2]*(p0[INDEX_3D_to_1D(iterx, itery+1,iterz)]-2.0*p0[INDEX_3D_to_1D(iterx, itery,iterz)]+p0[INDEX_3D_to_1D(iterx, itery-1,iterz)])
                 + coeff[3]*(p0[INDEX_3D_to_1D(iterx, itery,iterz+1)]-2.0*p0[INDEX_3D_to_1D(iterx, itery,iterz)]+p0[INDEX_3D_to_1D(iterx, itery,iterz-1)]);
	  pTAp_local += p0[INDEX_3D_to_1D(iterx,itery,iterz)]*Ap[INDEX_3D_to_1D(iterx, itery,iterz)];
        }
      }
    }
    double pTAp; 
    MPI_Allreduce(&pTAp_local, &pTAp, 1, MPI_DOUBLE, MPI_SUM, mycomm);
    double alpha = rTr_prev/pTAp;
    double rTr_local = 0.0;
    for (iterz=zid_start; iterz<zid_end+1; iterz++) {
      for (itery=yid_start; itery<yid_end+1; itery++) {
        for (iterx=xid_start; iterx<xid_end+1; iterx++) {
          u[INDEX_3D_to_1D(iterx, itery,iterz)]  += alpha*p0[INDEX_3D_to_1D(iterx, itery,iterz)];
          r0[INDEX_3D_to_1D(iterx, itery,iterz)] -=alpha*Ap[INDEX_3D_to_1D(iterx, itery,iterz)];
          rTr_local                              += r0[INDEX_3D_to_1D(iterx, itery,iterz)]*r0[INDEX_3D_to_1D(iterx, itery,iterz)] ;
        }
      }
    } 
    double localerror = absmax(r0);
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

    for (iterz=zid_start; iterz<zid_end+1; iterz++) {
      for (itery=yid_start; itery<yid_end+1; itery++) {
        for (iterx=xid_start; iterx<xid_end+1; iterx++) {
          p0[INDEX_3D_to_1D(iterx,itery,iterz)] = r0[INDEX_3D_to_1D(iterx, itery,iterz)]+ beta*p0[INDEX_3D_to_1D(iterx, itery,iterz)];
        }
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
  int NZ = atoi(argv[4]);
  double *u, *u_exact;
  int it, isconvergent;
  double diffnorm;
  double eps=2.e-13;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  ///////////////////////////////////////////////////////////////////////////////////////
  int periods[ndims];
  const int reorder=false;
  dims[0] = dims[1] = dims[2] = 0;
  MPI_Dims_create(nprocs, ndims, dims);
  if (rank==0 & 1==2)
  {
    printf("%d=%dx%dx%d.\n",nprocs,dims[0],dims[1],dims[2]);
  }

  periods[0] = periods[1] = periods[2]= false;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &mycomm);
  ///////////////////////////////////////////////////////////////////////////////////////
  int coords[ndims];
  MPI_Cart_coords(mycomm, rank, ndims, coords);
  xid_start = (coords[0]*NX)/dims[0]+1;
  xid_end = ((coords[0]+1)*NX)/dims[0];
  yid_start = (coords[1]*NY)/dims[1]+1;
  yid_end = ((coords[1]+1)*NY)/dims[1];
  zid_start = (coords[2]*NZ)/dims[2]+1;
  zid_end = ((coords[2]+1)*NZ)/dims[2];
  ///////////////////////////////////////////////////////////////////////////////////////

  int iterx, itery, iterz;
  double x, y, z;
  u = calloc( (xid_end-xid_start+3) * (yid_end-yid_start+3) * (zid_end-zid_start+3), sizeof(double));
  u_exact = calloc( (xid_end-xid_start+3) * (yid_end-yid_start+3) * (zid_end-zid_start+3), sizeof(double));
  f = calloc( (xid_end-xid_start+3) * (yid_end-yid_start+3) * (zid_end-zid_start+3), sizeof(double));
  dx = (b1-a1)/(NX+1.);
  dy = (b2-a2)/(NY+1.);
  dz = (b3-a3)/(NZ+1.);

  coeff[0] = (0.5*dx*dx*dy*dy*dz*dz)/(dx*dx*dy*dy+dy*dy*dz*dz+dz*dz*dx*dx);
  coeff[1] = 1./(dx*dx);
  coeff[2] = 1./(dy*dy);
  coeff[3] = 1./(dz*dz);

  for (iterz=zid_start-1; iterz<zid_end+2; iterz++) {
    for (itery=yid_start-1; itery<yid_end+2; itery++) {
      for(iterx=xid_start-1; iterx<xid_end+2; iterx++) {
        x = iterx*dx;
        y = itery*dy;
        z = iterz*dz;
        // Compute the source
        f[INDEX_3D_to_1D(iterx, itery,iterz)] = Source(x,y,z);
        // Compute the exact solution
        u_exact[INDEX_3D_to_1D(iterx, itery,iterz)] = exactsol(x,y,z);
        // Apply the boundary conditions
        if (iterx==0 || iterx==NX+1 || itery==0 || itery==NY+1 || iterz==0 || iterz==NZ+1)
        {
	  u[INDEX_3D_to_1D(iterx, itery,iterz)] = exactsol(x,y,z);
        }
      }
    }
  }
  ///////////////////////////////////////////////////////////////////////////
  MPI_Cart_shift(mycomm, 0, 1, &(neighbour[le]), &(neighbour[ri]));
  MPI_Cart_shift(mycomm, 1, 1, &(neighbour[fr]), &(neighbour[be]));
  MPI_Cart_shift(mycomm, 2, 1, &(neighbour[bo]), &(neighbour[to]));
  //printf("Process %d has six neighbours : left %d behind %d right %d front %d bottom %d top %d\n", rank, neighbour[le], neighbour[be], neighbour[ri], neighbour[fr],neighbour[bo],neighbour[to]);
  //printf("Process %d manages indices from %d to %d in x-direction, from %d to %d in y-direction and from %d to %d in z-direction.\n",rank,xid_start,xid_end,yid_start,yid_end,zid_start,zid_end);
  ///////////////////////////////////////////////////////////////////////////
  //MPI_Type_contiguous( (xid_end-xid_start+2)*(yid_end-yid_start+2), MPI_DOUBLE, &xyslice_type);
  //MPI_Type_commit(&xyslice_type);

  MPI_Type_vector( yid_end-yid_start+1, xid_end-xid_start+1, xid_end-xid_start+3 , MPI_DOUBLE, &xyslice_type);
  MPI_Type_commit(&xyslice_type);

  MPI_Type_vector( (yid_end-yid_start+3)*(zid_end-zid_start+3), 1, xid_end-xid_start+3, MPI_DOUBLE, &yzslice_type);
  MPI_Type_commit(&yzslice_type);

  MPI_Type_vector(zid_end-zid_start+1,xid_end-xid_start+1, (xid_end-xid_start+3)*(yid_end-yid_start+3), MPI_DOUBLE, &zxslice_type);
  MPI_Type_commit(&zxslice_type);

  double tstart = MPI_Wtime();
  if (solver[0]=='j' | solver[0]=='J')
  {
    jacobi3d_mpi(u,&it,eps);
  }
  else if (solver[0]=='c' | solver[0]=='C')
  {
    cg3d_mpi(u, &it, eps);
  }
  else
  {
    printf("Only Jacobi (J) and conjugate gradient (C) are available!\n");
    MPI_Abort(mycomm, 0);
  }
  
  double tend = MPI_Wtime();
  double error = max_error(u, u_exact);
  if (rank == 0 & 1==2) {
    if (it<MAX_ITERATION)
      printf("Converged after %d iterations in %f seconds.\n", it, tend-tstart);
      
    else
	printf("The solver reached the maximum number of iterations.\n");
    printf("Maximum error = %.1e\n",error);
  }
  if (rank==0)
  printf("%d %f %.1e\n",nprocs, tend-tstart,error);

  free(u);
  free(u_exact);
  free(f);
  MPI_Type_free (&xyslice_type);
  MPI_Type_free (&yzslice_type);
  MPI_Type_free (&zxslice_type); 
  MPI_Finalize();
  return 0;
}
