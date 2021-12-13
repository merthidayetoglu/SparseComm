#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

//GEOMETRY
char *filename;
int numcol;
int numrow;
long *displ;
int *index;
double *value;
int numrhs;
int numiter;

//TOPOLOGY
int myrank;
int numrank;
int numthread;

int myrank_node;
int numrank_node;

int myrank_node_t;
int numrank_node_t;

MPI_Comm MPI_COMM_NODE;
MPI_Comm MPI_COMM_NODE_T;

int main(int argc, char** argv) {

  double timetotal = MPI_Wtime();

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&numrank);

  #pragma omp parallel
  if(omp_get_thread_num() == 0)
    numthread = omp_get_num_threads();

  //REPORT INPUT PARAMETERS
  filename = getenv("FILENAME");
  char *chartemp = getenv("NUMRHS");
  numrhs = atoi(chartemp);
  chartemp = getenv("PERNODE");
  int pernode = atoi(chartemp);
  chartemp = getenv("NUMITER");
  numiter = atoi(chartemp);
  MPI_Comm_split(MPI_COMM_WORLD,myrank/pernode,myrank,&MPI_COMM_NODE);
  MPI_Comm_rank(MPI_COMM_NODE,&myrank_node);
  MPI_Comm_size(MPI_COMM_NODE,&numrank_node);

  MPI_Comm_split(MPI_COMM_WORLD,myrank%pernode,myrank,&MPI_COMM_NODE_T);
  MPI_Comm_rank(MPI_COMM_NODE_T,&myrank_node_t);
  MPI_Comm_size(MPI_COMM_NODE_T,&numrank_node_t);

  printf("myrank: %d/%d myrank_node_t: %d/%d myrank_node: %d/%d\n",myrank,numrank,myrank_node_t,numrank_node_t,myrank_node,numrank_node);

  if(myrank == 0){
    printf("\nNUMBER OF PROCESSES: %d\n",numrank);
    printf("NUMBER OF NODES: %d\n",numrank_node_t);
    printf("PROCESSES PER NODE: %d\n",numrank_node);
    printf("NUMBER OF THREADS: %d\n\n",numthread);
    printf("FILE NAME: %s\n",filename);
    printf("NUMBER OF RHS: %d\n",numrhs);
    printf("NUMBER OF ITERATIONS: %d\n\n",numiter);
  }

  //READ INPUT MATRIX
  FILE *inputf = fopen(filename,"rb");
  fread(&numrow,sizeof(int),1,inputf);
  fread(&numcol,sizeof(int),1,inputf);
  if(myrank == 0){
    printf("NUMBER OF ROWS: %d x %d (%f GB)\n",numrow,numrhs,numrow*sizeof(double)/1.e9*numrhs);
    printf("NUMBER OF COLUMNS: %d x %d (%f GB)\n\n",numcol,numrhs,numcol*sizeof(double)/1.e9*numrhs);
  }
  int mynumrow = numrow/numrank;
  if(myrank < numrow%numrank) mynumrow++;
  int mynumcol = numcol/numrank;
  if(myrank < numcol%numrank) mynumcol++;
  int numrows[numrank];
  int numcols[numrank];
  MPI_Allgather(&mynumrow,1,MPI_INT,numrows,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&mynumcol,1,MPI_INT,numcols,1,MPI_INT,MPI_COMM_WORLD);
  int rowdispl[numrank+1];
  int coldispl[numrank+1];
  rowdispl[0] = 0;
  coldispl[0] = 0;
  for(int rank = 0; rank < numrank; rank++){
    rowdispl[rank+1] = rowdispl[rank] + numrows[rank];
    coldispl[rank+1] = coldispl[rank] + numcols[rank];
  }
  if(myrank == 0)
    for(int rank = 0; rank < numrank; rank++)
      printf("PROCESS: %d ROWS: %d x %d (%f GB) COLS: %d x %d (%f GB)\n",rank,numrows[rank],numrhs,numrows[rank]*sizeof(double)/1.e9*numrhs,numcols[rank],numrhs,numcols[rank]*sizeof(double)/1.e9*numrhs);

  //READ THE SPARSE MATRIX IN PARALLEL
  {
    MPI_Barrier(MPI_COMM_WORLD);
    double time = omp_get_wtime();
    displ = new long[mynumrow+1];
    fseek(inputf,sizeof(long)*rowdispl[myrank],SEEK_CUR);
    fread(displ,sizeof(long),mynumrow+1,inputf);
    long strind = displ[0];
    #pragma omp parallel for
    for(int m = 0; m < mynumrow+1; m++)
     displ[m] -= strind;
    long numnztot = displ[mynumrow];
    long numnzall[numrank];
    MPI_Allgather(&numnztot,1,MPI_LONG,numnzall,1,MPI_LONG,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&numnztot,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    if(myrank == 0){
      for(int rank = 0; rank < numrank; rank++)
        printf("PROCESS: %d NONZEROES: %ld (%f GB)\n",rank,numnzall[rank],numnzall[rank]*(sizeof(int)+sizeof(double))/1.e9);
      printf("NUMBER OF NONZEROES: %ld (%f GB)\n",numnztot,numnztot*(sizeof(int)+sizeof(double))/1.e9);
    }
    fseek(inputf,sizeof(long)*(numrow+1-rowdispl[myrank]),SEEK_CUR);
    fseek(inputf,sizeof(int)*strind,SEEK_CUR);
    index = new int[displ[mynumrow]];
    fread(index,sizeof(int),displ[mynumrow],inputf);
    fclose(inputf);
    value = new double[displ[mynumrow]];
    #pragma omp parallel for
    for(long n = 0; n < displ[mynumrow]; n++)
      value[n] = n;
    MPI_Barrier(MPI_COMM_WORLD);
    if(myrank == 0)printf("I/O TIME: %f\n\n",omp_get_wtime()-time);
  }
  
  double *B = new double[mynumcol*numrhs];
  #pragma omp parallel for
  for(int n = 0; n < mynumcol*numrhs; n++)
    B[n] = n;
  double *C = new double[mynumrow*numrhs];

  //PERFORM DENSE COMMUNICATIONS
  {
    if(myrank == 0){
      printf("DENSE COMMUNICATIONS\n");
      for(int rank = 0; rank < numrank; rank++)
        printf("PROCESS %d SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n",rank,numcols[rank]*numrank,numrhs,numcols[rank]*numrank*sizeof(double)/1.e9*numrhs,numcol,numrhs,numcol*sizeof(double)/1.e9*numrhs);
      printf("TOTAL COMM SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n\n",numcol*numrank,numrhs,numcol*numrank*sizeof(double)/1.e9*numrhs,numcol*numrank,numrhs,numcol*numrank*sizeof(double)/1.e9*numrhs);
    }
    double *recvbuff = new double[numcol*numrhs];
    double *unpackbuff = new double[numcol*numrhs];
    int numcolsdense[numrank];
    for(int rank = 0; rank < numrank; rank++)
      numcolsdense[rank] = numcols[rank]*numrhs;
    int coldispldense[numrank+1];
    for(int rank = 0; rank < numrank+1; rank++)
      coldispldense[rank] = coldispl[rank]*numrhs;
    int *unpackindex = new int[numcol*numrhs];
    for(int rank = 0; rank < numrank; rank++)
      #pragma omp parallel for
      for(int m = coldispl[rank]; m < coldispl[rank+1]; m++)
        for(int k = 0; k < numrhs; k++)
          unpackindex[k*numcol+m] = coldispldense[rank]+k*numcols[rank]+m-coldispl[rank];
    /*for(int k = 0; k < numrhs; k++){
      MPI_Allgatherv(B+k*mynumcol,mynumcol,MPI_DOUBLE,recvbuff,numcols,coldispl,MPI_DOUBLE,MPI_COMM_WORLD);
      for(int m = 0; m < mynumrow; m++){
        double reduce = 0;
	for(long n = displ[m]; n < displ[m+1]; n++)
	  reduce += recvbuff[index[n]]*value[n];
	C[k*mynumrow+m] = reduce;
      }
    }*/
    for(int iter = 0; iter < numiter; iter++){
      MPI_Barrier(MPI_COMM_WORLD);
      double timecomm = omp_get_wtime();
      MPI_Allgatherv(B,mynumcol*numrhs,MPI_DOUBLE,recvbuff,numcolsdense,coldispldense,MPI_DOUBLE,MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      timecomm = omp_get_wtime()-timecomm;
      double timepack = omp_get_wtime();
      #pragma omp parallel for
      for(int n = 0; n < numcol*numrhs; n++)
        unpackbuff[n] = recvbuff[unpackindex[n]];
      MPI_Barrier(MPI_COMM_WORLD);
      timepack = omp_get_wtime()-timepack;
      double timekernel = omp_get_wtime();
      #pragma omp parallel for
      for(int m = 0; m < mynumrow; m++){
        double reduce[numrhs] = {0};
        for(long n = displ[m]; n < displ[m+1]; n++){
          int ind = index[n];
          double val = value[n]; 
          for(int k = 0; k < numrhs; k++)
            reduce[k] += unpackbuff[k*numcol+ind]*val;
        }
        for(int k = 0; k < numrhs; k++)
          C[k*mynumrow+m] = reduce[k];
      }
      MPI_Barrier(MPI_COMM_WORLD);
      timekernel = omp_get_wtime()-timekernel;
      if(myrank == 0)printf("timecomm: %e (%f GB/s) timepack: %e timekernel: %e\n",timecomm,numcol*sizeof(double)/1.e9*numrank/timecomm*numrhs,timepack,timekernel);
    }
    delete[] recvbuff;
    delete[] unpackbuff;
    delete[] unpackindex;
  }
  double *Cdense = C;
  C = new double[mynumrow*numrhs];
  //PERFORM SPARSE COMM
  {
    //FIND COMMUNICATION FOOTPRINT
    int *footprint = new int[numcol];
    #pragma omp parallel for
    for(int n = 0; n < numcol; n++)
      footprint[n] = 0;
    for(int m = 0; m < mynumrow; m++)
      for(long n = displ[m]; n < displ[m+1]; n++)
        footprint[index[n]] = 1;
    int recvcount[numrank] = {0};
    #pragma omp parallel for
    for(int rank = 0; rank < numrank; rank++)
      for(int n = coldispl[rank]; n < coldispl[rank+1]; n++)
        if(footprint[n] > 0)
          recvcount[rank]++;
    int sendcount[numrank];
    MPI_Alltoall(recvcount,1,MPI_INT,sendcount,1,MPI_INT,MPI_COMM_WORLD);
    int recvdispl[numrank+1] = {0};
    int senddispl[numrank+1] = {0};
    for(int rank = 0; rank < numrank; rank++){
      recvdispl[rank+1] = recvdispl[rank] + recvcount[rank];
      senddispl[rank+1] = senddispl[rank] + sendcount[rank];
    }
    int sendtot = senddispl[numrank];
    int recvtot = recvdispl[numrank];
    int sendall[numrank];
    int recvall[numrank];
    MPI_Allgather(&sendtot,1,MPI_INT,sendall,1,MPI_INT,MPI_COMM_WORLD);
    MPI_Allgather(&recvtot,1,MPI_INT,recvall,1,MPI_INT,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&sendtot,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&recvtot,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(myrank == 0){
      printf("\nSPARSE COMMUNICATIONS\n");
      for(int rank = 0; rank < numrank; rank++)
        printf("PROCESS %d SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n",rank,sendall[rank],numrhs,sendall[rank]*sizeof(double)/1.e9*numrhs,recvall[rank],numrhs,recvall[rank]*sizeof(double)/1.e9*numrhs);
      printf("TOTAL COMM SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n\n",sendtot,numrhs,sendtot*sizeof(double)/1.e9*numrhs,recvtot,numrhs,recvtot*sizeof(double)/1.e9*numrhs);
    }
    //CONSTRUCT SEND & RECEIVE INDICES
    int *recvindex = new int[recvdispl[numrank]];
    int *sendindex = new int[senddispl[numrank]];
    #pragma omp parallel for
    for(int rank = 0; rank < numrank; rank++){
      int count = 0;
      for(int n = coldispl[rank]; n < coldispl[rank+1]; n++)
        if(footprint[n] > 0){
          int index = recvdispl[rank]+count;
          recvindex[index] = n-coldispl[rank];
          footprint[n] = index;
          count++;
        }
    }
    MPI_Alltoallv(recvindex,recvcount,recvdispl,MPI_INT,sendindex,sendcount,senddispl,MPI_INT,MPI_COMM_WORLD);
    //UPDATE INDEX VALUES
    int *index_temp = new int[displ[mynumrow]];
    #pragma omp parallel for
    for(int m = 0; m < mynumrow; m++)
      for(long n = displ[m]; n < displ[m+1]; n++)
        index_temp[n] = footprint[index[n]];

    double *recvbuff = new double[recvdispl[numrank]*numrhs];
    double *sendbuff = new double[senddispl[numrank]*numrhs];
    int sendcountsparse[numrank];
    int senddisplsparse[numrank+1];
    int recvcountsparse[numrank];
    int recvdisplsparse[numrank+1];
    for(int rank = 0; rank < numrank; rank++){
      sendcountsparse[rank] = sendcount[rank]*numrhs;
      recvcountsparse[rank] = recvcount[rank]*numrhs;
    }
    for(int rank = 0; rank < numrank+1; rank++){
      senddisplsparse[rank] = senddispl[rank]*numrhs;
      recvdisplsparse[rank] = recvdispl[rank]*numrhs;
    }
    int *packindex = new int[senddisplsparse[numrank]];
    int *unpackindex = new int[recvdisplsparse[numrank]];
    for(int rank = 0; rank < numrank; rank++)
      #pragma omp parallel for
      for(int n = senddispl[rank]; n < senddispl[rank+1]; n++)
        for(int k = 0; k < numrhs; k++)
          packindex[senddisplsparse[rank]+k*sendcount[rank]+n-senddispl[rank]] = k*mynumcol+sendindex[n];
    for(int rank = 0; rank < numrank; rank++)
      #pragma omp parallel for
      for(int m = recvdispl[rank]; m < recvdispl[rank+1]; m++)
        for(int k = 0; k < numrhs; k++)
          unpackindex[k*recvdispl[numrank]+m] = recvdisplsparse[rank]+k*recvcount[rank]+m-recvdispl[rank];
    double *unpackbuff = new double[recvdisplsparse[numrank]];

    /*for(int k = 0; k < numrhs; k++){
      for(int n = 0; n < senddispl[numrank]; n++)
        sendbuff[n] = sendindex[n];
      MPI_Alltoallv(sendbuff,sendcount,senddispl,MPI_DOUBLE,recvbuff,recvcount,recvdispl,MPI_DOUBLE,MPI_COMM_WORLD);
      for(int m = 0; m < mynumrow; m++){
        double reduce = 0;
	for(long n = displ[m]; n < displ[m+1]; n++)
          reduce += recvbuff[index[n]]*value[n];
        C[k*mynumrow+m] = reduce;
      }
    }*/

    for(int iter = 0; iter < numiter; iter++){
      MPI_Barrier(MPI_COMM_WORLD);
      double timepack = omp_get_wtime();
      #pragma omp parallel for
      for(int n = 0; n < senddispl[numrank]*numrhs; n++)
        sendbuff[n] = B[packindex[n]];
      MPI_Barrier(MPI_COMM_WORLD);
      timepack = omp_get_wtime()-timepack;
      double timecomm = omp_get_wtime();
      MPI_Alltoallv(sendbuff,sendcountsparse,senddisplsparse,MPI_DOUBLE,recvbuff,recvcountsparse,recvdisplsparse,MPI_DOUBLE,MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      timecomm = omp_get_wtime()-timecomm;
      double timeunpack = omp_get_wtime();
      #pragma omp parallel for
      for(int n = 0; n < recvdispl[numrank]*numrhs; n++)
        unpackbuff[n] = recvbuff[unpackindex[n]];
      MPI_Barrier(MPI_COMM_WORLD);
      timeunpack  = omp_get_wtime()-timeunpack;
      double timekernel = omp_get_wtime();
      #pragma omp parallel for
      for(int m = 0; m < mynumrow; m++){
        double reduce[numrhs] = {0};
        for(long n = displ[m]; n < displ[m+1]; n++){
          int ind = index_temp[n];
          double val = value[n];
          for(int k = 0; k < numrhs; k++)
            reduce[k] += unpackbuff[k*recvdispl[numrank]+ind]*val;
        }
        for(int k = 0; k < numrhs; k++)
          C[k*mynumrow+m] = reduce[k];
      }
      MPI_Barrier(MPI_COMM_WORLD);
      timekernel = omp_get_wtime()-timekernel;
      if(myrank == 0)printf("timepack: %e timecomm: %e (%f GB/s) timeunpack: %e timekernel: %e\n",timepack,timecomm,sendtot*sizeof(double)/1.e9/timecomm*numrhs,timeunpack,timekernel);
    }
    delete[] sendbuff;
    delete[] recvbuff;
    delete[] packindex;
    delete[] unpackindex;
    delete[] unpackbuff;
  }
  for(int m = 0; m < mynumrow*numrhs; m++)
    if(Cdense[m] != C[m]){
      printf("%e %e LAN!!!!!!!!!!!!!!!!!!!!!!!\n",Cdense[m],C[m]);
      break;
    }
  delete[] C;
  C = new double[mynumrow*numrhs];
  //PERFORM HIERARCHICAL COMM
  {
    //FIND COMMUNICATION FOOTPRINT OF EACH NODE
    int *footprint_node = new int[numcol];
    #pragma omp parallel for
    for(int n = 0; n < numcol; n++)
      footprint_node[n] = 0;
    for(int m = 0; m < mynumrow; m++)
      for(long n = displ[m]; n < displ[m+1]; n++)
        footprint_node[index[n]] = 1;
    MPI_Allreduce(MPI_IN_PLACE,footprint_node,numcol,MPI_INT,MPI_SUM,MPI_COMM_NODE);
    int recvcount_node[numrank_node_t] = {0};
    #pragma omp parallel for
    for(int node = 0; node < numrank_node_t; node++){
      int rank = node*numrank_node+myrank_node;
      for(int m = coldispl[rank]; m < coldispl[rank+1]; m++)
        if(footprint_node[m] > 0)
          recvcount_node[node]++;
    }
    int sendcount_node[numrank_node_t];
    MPI_Alltoall(recvcount_node,1,MPI_INT,sendcount_node,1,MPI_INT,MPI_COMM_NODE_T);
    int recvdispl_node[numrank_node_t+1] = {0};
    int senddispl_node[numrank_node_t+1] = {0};
    for(int node = 0; node < numrank_node_t; node++){
      recvdispl_node[node+1] = recvdispl_node[node] + recvcount_node[node];
      senddispl_node[node+1] = senddispl_node[node] + sendcount_node[node];
    }
    int sendtot_node = senddispl_node[numrank_node_t];
    int recvtot_node = recvdispl_node[numrank_node_t];
    int sendall_node[numrank];
    int recvall_node[numrank];
    MPI_Allgather(&sendtot_node,1,MPI_INT,sendall_node,1,MPI_INT,MPI_COMM_WORLD);
    MPI_Allgather(&recvtot_node,1,MPI_INT,recvall_node,1,MPI_INT,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&sendtot_node,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&recvtot_node,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(myrank == 0){
      printf("\nGLOBAL COMMUNICATIONS\n");
      for(int rank = 0; rank < numrank; rank++)
        printf("PROCESS %d SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n",rank,sendall_node[rank],numrhs,sendall_node[rank]*sizeof(double)/1.e9*numrhs,recvall_node[rank],numrhs,recvall_node[rank]*sizeof(double)/1.e9*numrhs);
      printf("TOTAL COMM SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n\n",sendtot_node,numrhs,sendtot_node*sizeof(double)/1.e9*numrhs,recvtot_node,numrhs,recvtot_node*sizeof(double)/1.e9*numrhs);
    }
    //CONSTRUCT SEND & RECEIVE INDICES
    int *recvindex_node = new int[recvdispl_node[numrank_node_t]];
    int *sendindex_node = new int[senddispl_node[numrank_node_t]];
    #pragma omp parallel for
    for(int node = 0; node < numrank_node_t; node++){
      int rank = node*numrank_node+myrank_node;
      int count = 0;
      for(int m = coldispl[rank]; m < coldispl[rank+1]; m++)
        if(footprint_node[m] > 0){
          int index = recvdispl_node[node]+count;
          recvindex_node[index] = m-coldispl[rank];
          footprint_node[m] = index;
          count++;
        }
    }
    MPI_Alltoallv(recvindex_node,recvcount_node,recvdispl_node,MPI_INT,sendindex_node,sendcount_node,senddispl_node,MPI_INT,MPI_COMM_NODE_T);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if(myrank == 0)printf("\nTOTAL TIME: %e\n\n",MPI_Wtime()-timetotal);

  MPI_Finalize();

  return 0;
}
