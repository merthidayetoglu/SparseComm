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

//TOPOLOGY
int myrank;
int numrank;
int numthread;

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
  if(myrank == 0){
    printf("\nNUMBER OF PROCESSES: %d\n",numrank);
    printf("NUMBER OF THREADS: %d\n\n",numthread);
    printf("FILE NAME: %s\n",filename);
    printf("NUMBER OF RHS: %d\n\n",numrhs);
  }

  //READ INPUT MATRIX
  FILE *inputf = fopen(filename,"rb");
  fread(&numrow,sizeof(int),1,inputf);
  fread(&numcol,sizeof(int),1,inputf);
  if(myrank == 0){
    printf("NUMBER OF ROWS: %d (%f MB)\n",numrow,numrow*sizeof(double)/1.e6);
    printf("NUMBER OF COLUMNS: %d (%f MB)\n",numcol,numcol*sizeof(double)/1.e6);
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
      printf("PROCESS: %d ROWS: %d COLS: %d\n",rank,numrows[rank],numcols[rank]);

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
      value[n] = 1.0;
    MPI_Barrier(MPI_COMM_WORLD);
    if(myrank == 0)printf("I/O TIME: %f\n\n",omp_get_wtime()-time);
  }

  //FIGURE OUT MEMORY FOOTPRINT
  int *footprint = new int[numcol];
  #pragma omp parallel for
  for(int n = 0; n < numcol; n++)
    footprint[n] = -1;
  for(int m = 0; m < mynumrow; m++)
    for(long n = displ[m]; n < displ[m+1]; n++)
      footprint[index[n]] = 1;
  int recvcount[numrank] = {0};
  #pragma omp parallel for
  for(int rank = 0; rank < numrank; rank++)
    for(int n = coldispl[rank]; n < coldispl[rank+1]; n++)
      if(footprint[n] > -1)
        recvcount[rank]++;
  int sendcount[numrank];
  MPI_Alltoall(recvcount,1,MPI_INT,sendcount,1,MPI_INT,MPI_COMM_WORLD);
  int recvdispl[numrank+1] = {0};
  int senddispl[numrank+1] = {0};
  for(int rank = 0; rank < numrank; rank++){
    recvdispl[rank+1] = recvdispl[rank] + recvcount[rank];
    senddispl[rank+1] = senddispl[rank] + sendcount[rank];
  }
  long sendtot = senddispl[numrank];
  long recvtot = recvdispl[numrank];
  MPI_Allreduce(MPI_IN_PLACE,&sendtot,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&recvtot,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  if(myrank == 0){
    for(int rank = 0; rank < numrank; rank++)
      printf("send %d recv %d\n",sendcount[rank],recvcount[rank]);
    printf("TOTAL COMM SEND %ld (%f MB) RECV %ld (%f MB)\n\n",sendtot,sendtot*sizeof(double)/1.e6,recvtot,recvtot*sizeof(double)/1.e6);
  }

  int *recvindex = new int[recvdispl[numrank]];
  #pragma omp parallel for
  for(int rank = 0; rank < numrank; rank++){
    int count = 0;
    for(int n = coldispl[rank]; n < coldispl[rank+1]; n++)
      if(footprint[n] > -1){
	int index = recvdispl[rank]+count;
        recvindex[index] = n-coldispl[rank];
	footprint[n] = index;
	count++;
      }
  }
  int *sendindex = new int[senddispl[numrank]];
  MPI_Alltoallv(recvindex,recvcount,recvdispl,MPI_INT,sendindex,sendcount,senddispl,MPI_INT,MPI_COMM_WORLD);

  double *B = new double[mynumcol];
  #pragma omp parallel for
  for(int n = 0; n < mynumcol; n++)
    B[n] = 1;
  double *C = new double[mynumrow];

  //PERFORM DENSE COMM
  {
    double *recvbuff = new double[numcol];
    MPI_Barrier(MPI_COMM_WORLD);
    double timecomm = omp_get_wtime();
    MPI_Allgatherv(B,mynumcol,MPI_DOUBLE,recvbuff,numcols,coldispl,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    timecomm = omp_get_wtime()-timecomm;
    double timekernel = omp_get_wtime();
    #pragma omp parallel for
    for(int m = 0; m < mynumrow; m++){
      double reduce = 0;
      for(long n = displ[m]; n < displ[m+1]; n++)
        reduce += recvbuff[index[n]];
      C[m] = reduce;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    timekernel = omp_get_wtime()-timekernel;
    if(myrank == 0)printf("timecomm: %e timekernel: %e\n",timecomm,timekernel);
    delete[] recvbuff;
  }
  double *Cdense = C;
  C = new double[mynumrow];
  double *recvbuff = new double[recvdispl[numrank]];
  double *sendbuff = new double[senddispl[numrank]];
  #pragma omp parallel for
  for(int m = 0; m < mynumrow; m++)
    for(long n = displ[m]; n < displ[m+1]; n++)
      index[n] = footprint[index[n]];
  //PERFORM SPARSE COMM
  {
    MPI_Barrier(MPI_COMM_WORLD);
    double timereduce = omp_get_wtime();
    #pragma omp parallel for
    for(int n = 0; n < senddispl[numrank]; n++)
      sendbuff[n] = B[sendindex[n]];
    MPI_Barrier(MPI_COMM_WORLD);
    timereduce = omp_get_wtime()-timereduce;
    double timecomm = omp_get_wtime();
    MPI_Alltoallv(sendbuff,sendcount,senddispl,MPI_DOUBLE,recvbuff,recvcount,recvdispl,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    timecomm = omp_get_wtime()-timecomm;
    double timekernel = omp_get_wtime();
    #pragma omp parallel for
    for(int m = 0; m < mynumrow; m++){
      double reduce = 0;
      for(long n = displ[m]; n < displ[m+1]; n++)
        reduce += recvbuff[index[n]];
      C[m] = reduce;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    timekernel = omp_get_wtime()-timekernel;
    if(myrank == 0)printf("timereduce: %e timecomm: %e timekernel: %e\n",timereduce,timecomm,timekernel);
  }

  double *Csparse = C;
  for(int m = 0; m < mynumrow; m++)
    if(Cdense[m] != Csparse[m]){
      printf("LAN!\n");
      break;
    }


  //PRINT COMMUNICATION MAP
  /*int commap[numrank*numrank];
  MPI_Allgather(recvcount,numrank,MPI_INT,commap,numrank,MPI_INT,MPI_COMM_WORLD);
  if(myrank == 0)
    for(int m = 0; m < numrank; m++){
      for(int n = 0; n < numrank; n++)
        printf("%d ",commap[m*numrank+n]);
      printf("\n");
    }
  if(myrank == 2){
    printf("process %d sends",myrank);
    for(int rank = 0; rank < numrank; rank++)
      printf(" %d",sendcount[rank]);
    printf("\n");
    printf("process %d recvs",myrank);
    for(int rank = 0; rank < numrank; rank++)
      printf(" %d",recvcount[rank]);
    printf("\n\n");
  }*/

  /*int buffsize = 0;
  for(int n = 0; n < numcol; n++)
    if(footprint[n] > 0){
      footprint[n] = buffsize;
      buffsize++;
    }
  //REINDEXING
  #pragma omp parallel for
  for(int m = 0; m < mynumrow; m++)
    for(long n = displ[m]; n < displ[m+1]; n++)
      index[n] = footprint[index[n]];
  printf("proc %d buffsize %d\n",myrank,buffsize);*/



  MPI_Barrier(MPI_COMM_WORLD);
  if(myrank == 0)printf("TOTAL TIME: %e\n",MPI_Wtime()-timetotal);

  MPI_Finalize();

  return 0;
}