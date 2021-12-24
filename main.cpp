#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define PRINTID 0

//GEOMETRY
char *filename;
long *csrdispl;
int *csrindex;
double *csrvalue;
int numrhs;
int numiter;

//TOPOLOGY
int myid;
int numproc;
int myid_node;
int numproc_node;
int mynode;
int numnode;
int myid_socket;
int numproc_socket;
int mysocket;
int numsocket;
int numthread;

MPI_Comm MPI_COMM_NODE;
MPI_Comm MPI_COMM_NODE_T;
MPI_Comm MPI_COMM_SOCKET;
MPI_Comm MPI_COMM_SOCKET_T;

int main(int argc, char** argv) {

  double timetotal = omp_get_wtime();

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);

  #pragma omp parallel
  if(omp_get_thread_num() == 0)
    numthread = omp_get_num_threads();

  //REPORT INPUT PARAMETERS
  filename = getenv("FILENAME");
  char *chartemp = getenv("NUMRHS");
  numrhs = atoi(chartemp);
  chartemp = getenv("PERNODE");
  int pernode = atoi(chartemp);
  chartemp = getenv("PERSOCKET");
  int persocket = atoi(chartemp);
  chartemp = getenv("NUMITER");
  numiter = atoi(chartemp);

  if(numproc%pernode || pernode%persocket){
    printf("INCONSISTENT NUMBER OF PROCESSES PER NODE OR PER SOCKET: %d %d %d\n",numproc,pernode,persocket);
    return 0;
  }

  //WITHIN NODES
  MPI_Comm_split(MPI_COMM_WORLD,myid/pernode,myid,&MPI_COMM_NODE);
  MPI_Comm_rank(MPI_COMM_NODE,&myid_node);
  MPI_Comm_size(MPI_COMM_NODE,&numproc_node);
  //ACROSS NODES
  MPI_Comm_split(MPI_COMM_WORLD,myid%numproc_node,myid,&MPI_COMM_NODE_T);
  MPI_Comm_rank(MPI_COMM_NODE_T,&mynode);
  MPI_Comm_size(MPI_COMM_NODE_T,&numnode);
  //WITHIN SOCKETS
  MPI_Comm_split(MPI_COMM_NODE,myid_node/persocket,myid_node,&MPI_COMM_SOCKET);
  MPI_Comm_rank(MPI_COMM_SOCKET,&myid_socket);
  MPI_Comm_size(MPI_COMM_SOCKET,&numproc_socket);
  //ACROSS SOCKETS
  MPI_Comm_split(MPI_COMM_NODE,myid_node%numproc_socket,myid_node,&MPI_COMM_SOCKET_T);
  MPI_Comm_rank(MPI_COMM_SOCKET_T,&mysocket);
  MPI_Comm_size(MPI_COMM_SOCKET_T,&numsocket);

  int myid_all[numproc];
  int numproc_all[numproc];
  int myid_node_all[numproc];
  int numproc_node_all[numproc];
  int mynode_all[numproc];
  int numnode_all[numproc];
  int myid_socket_all[numproc];
  int numproc_socket_all[numproc];
  int mysocket_all[numproc];
  int numsocket_all[numproc];
  MPI_Allgather(&myid,1,MPI_INT,myid_all,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&numproc,1,MPI_INT,numproc_all,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&myid_node,1,MPI_INT,myid_node_all,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&numproc_node,1,MPI_INT,numproc_node_all,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&mynode,1,MPI_INT,mynode_all,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&numnode,1,MPI_INT,numnode_all,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&myid_socket,1,MPI_INT,myid_socket_all,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&numproc_socket,1,MPI_INT,numproc_socket_all,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&mysocket,1,MPI_INT,mysocket_all,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&numsocket,1,MPI_INT,numsocket_all,1,MPI_INT,MPI_COMM_WORLD);

  if(myid == PRINTID){
    printf("\n");
    for(int proc = 0; proc < numproc; proc++)
      printf("myid: %d/%d mynode: %d/%d myid_node: %d/%d mysocket %d/%d myid_socket %d/%d\n",myid_all[proc],numproc_all[proc],mynode_all[proc],numnode_all[proc],myid_node_all[proc],numproc_node_all[proc],mysocket_all[proc],numsocket_all[proc],myid_socket_all[proc],numproc_socket_all[proc]);
  }
  if(myid == PRINTID){
    printf("\nNUMBER OF PROCESSES: %d\n",numproc);
    printf("PROCESSES PER SOCKET: %d\n",numproc_socket);
    printf("PROCESSES PER NODE: %d\n",numproc_node);
    printf("NUMBER OF SOCKETS: %d\n",numsocket);
    printf("NUMBER OF NODES: %d\n\n",numnode);
    printf("NUMBER OF THREADS PER PROCESS: %d\n\n",numthread);
    printf("FILE NAME: %s\n",filename);
    printf("NUMBER OF RHS: %d\n",numrhs);
    printf("NUMBER OF ITERATIONS: %d\n\n",numiter);
  }

  //READ INPUT MATRIX
  int numrow;
  int numcol;
  FILE *inputf = fopen(filename,"rb");
  fread(&numrow,sizeof(int),1,inputf);
  fread(&numcol,sizeof(int),1,inputf);
  if(myid == PRINTID){
    printf("NUMBER OF ROWS: %d x %d (%f GB)\n",numrow,numrhs,numrow*sizeof(double)/1.e9*numrhs);
    printf("NUMBER OF COLUMNS: %d x %d (%f GB)\n\n",numcol,numrhs,numcol*sizeof(double)/1.e9*numrhs);
  }
  int mynumrow = numrow/numproc;
  if(myid < numrow%numproc) mynumrow++;
  int mynumcol = numcol/numproc;
  if(myid < numcol%numproc) mynumcol++;

  int numrows[numproc];
  int numcols[numproc];
  MPI_Allgather(&mynumrow,1,MPI_INT,numrows,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&mynumcol,1,MPI_INT,numcols,1,MPI_INT,MPI_COMM_WORLD);
  int rowdispl[numproc+1];
  int coldispl[numproc+1];
  rowdispl[0] = 0;
  coldispl[0] = 0;
  for(int proc = 0; proc < numproc; proc++){
    rowdispl[proc+1] = rowdispl[proc] + numrows[proc];
    coldispl[proc+1] = coldispl[proc] + numcols[proc];
  }
  if(myid == PRINTID)
    for(int proc = 0; proc < numproc; proc++)
      printf("PROCESS: %d ROWS: %d x %d (%f GB) COLS: %d x %d (%f GB)\n",myid,numrows[proc],numrhs,numrows[proc]*sizeof(double)/1.e9*numrhs,numcols[proc],numrhs,numcols[proc]*sizeof(double)/1.e9*numrhs);

  //READ THE SPARSE MATRIX IN PARALLEL
  {
    MPI_Barrier(MPI_COMM_WORLD);
    double time = omp_get_wtime();
    csrdispl = new long[mynumrow+1];
    fseek(inputf,sizeof(long)*rowdispl[myid],SEEK_CUR);
    fread(csrdispl,sizeof(long),mynumrow+1,inputf);
    long strind = csrdispl[0];
    for(int m = 0; m < mynumrow+1; m++)
      csrdispl[m] -= strind;
    long numnztot = csrdispl[mynumrow];
    long numnzall[numproc];
    MPI_Allgather(&numnztot,1,MPI_LONG,numnzall,1,MPI_LONG,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&numnztot,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    if(myid == PRINTID){
      for(int proc = 0; proc < numproc; proc++)
        printf("PROCESS: %d NONZEROES: %ld (%f GB)\n",proc,numnzall[proc],numnzall[proc]*(sizeof(int)+sizeof(double))/1.e9);
      printf("NUMBER OF NONZEROES: %ld (%f GB)\n",numnztot,numnztot*(sizeof(int)+sizeof(double))/1.e9);
    }
    fseek(inputf,sizeof(long)*(numrow-rowdispl[myid+1]),SEEK_CUR);
    fseek(inputf,sizeof(int)*strind,SEEK_CUR);
    csrindex = new int[csrdispl[mynumrow]];
    fread(csrindex,sizeof(int),csrdispl[mynumrow],inputf);
    fclose(inputf);
    csrvalue = new double[csrdispl[mynumrow]];
    for(long n = 0; n < csrdispl[mynumrow]; n++)
      csrvalue[n] = strind+n;
    MPI_Barrier(MPI_COMM_WORLD);
    if(myid == PRINTID)printf("I/O TIME: %f\n\n",omp_get_wtime()-time);
  }

  int sparsity[numrow*numcol];
  for(int n = 0; n < numrow*numcol; n++)
    sparsity[n] = 0;
  for(int m = 0; m < mynumrow; m++)
    for(int n = csrdispl[m]; n < csrdispl[m+1]; n++)
      sparsity[(rowdispl[myid]+m)*numcol+csrindex[n]] = 1;
  MPI_Allreduce(MPI_IN_PLACE,sparsity,numrow*numcol,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if(myid == PRINTID){
    printf("SPARSITY\n");
    for(int m = 0; m < numrow; m++){
      for(int n = 0; n < numcol; n++)
        printf("%d ",sparsity[m*numcol+n]);
      printf("\n");
    }
    printf("\n");
  }

  double *B = new double[mynumcol*numrhs];
  for(int m = 0; m < mynumcol; m++)
    for(int k = 0; k < numrhs; k++)
      B[k*mynumcol+m] = k*numcol+coldispl[myid]+m;
  double *C = new double[mynumrow*numrhs];

  //FIND COMMUNICATION FOOTPRINT OF EACH PROCESS
  bool *footprint = new bool[numcol];
  for(int n = 0; n < numcol; n++)
    footprint[n] = false;
  for(int m = 0; m < mynumrow; m++)
    for(long n = csrdispl[m]; n < csrdispl[m+1]; n++)
      footprint[csrindex[n]] = true;
  //FIND GLOBAL COMMUNICATIONS
  bool *footprint_node = new bool[numcol];
  MPI_Allreduce(footprint,footprint_node,numcol,MPI_C_BOOL,MPI_LOR,MPI_COMM_NODE);
  int sendcount_global[numnode];
  int recvcount_global[numnode];
  for(int node = 0; node < numnode; node++){
    int count = 0;
    int proc = node*numproc_node+myid_node;
    for(int m = coldispl[proc]; m < coldispl[proc+1]; m++)
      if(footprint_node[m])
        count++;
    recvcount_global[node] = count;
  }
  MPI_Alltoall(recvcount_global,1,MPI_INT,sendcount_global,1,MPI_INT,MPI_COMM_NODE_T);
  int senddispl_global[numnode+1];
  int recvdispl_global[numnode+1];
  senddispl_global[0] = 0;
  recvdispl_global[0] = 0;
  for(int node = 0; node < numnode; node++){
    senddispl_global[node+1] = senddispl_global[node] + sendcount_global[node];
    recvdispl_global[node+1] = recvdispl_global[node] + recvcount_global[node];
  }
  int commap_global[numproc*numproc];
  for(int m = 0; m < numproc*numproc; m++)
    commap_global[m] = 0;
  for(int node = 0; node < numnode; node++){
    int proc = node*numproc_node+myid_node;
    if(recvcount_global[node])
      commap_global[myid*numproc+proc] = recvcount_global[node];
  }
  MPI_Allreduce(MPI_IN_PLACE,commap_global,numproc*numproc,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  int sendall_global[numproc];
  int recvall_global[numproc];
  MPI_Allgather(&senddispl_global[numnode],1,MPI_INT,sendall_global,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&recvdispl_global[numnode],1,MPI_INT,recvall_global,1,MPI_INT,MPI_COMM_WORLD);
  if(myid == PRINTID){
    printf("\n");
    printf("GLOBAL (INTER-NODE) COMMUNICATIONS\n");
    int sendtot = 0;
    int recvtot = 0;
    for(int proc = 0; proc < numproc; proc++){
      printf("PROCESS %d SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n",proc,sendall_global[proc],numrhs,sendall_global[proc]*sizeof(double)/1.e9*numrhs,recvall_global[proc],numrhs,recvall_global[proc]*sizeof(double)/1.e9*numrhs);
      sendtot += sendall_global[proc];
      recvtot += recvall_global[proc];
    }
    printf("TOTAL COMM SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n",sendtot,numrhs,sendtot*sizeof(double)/1.e9*numrhs,recvtot,numrhs,recvtot*sizeof(double)/1.e9*numrhs);
    printf("COMM MAP:\n");
    for(int m = 0; m < numproc; m++){
      for(int n = 0; n < numproc; n++)
        printf("%d ",commap_global[m*numproc+n]);
      printf("\n");
    }
  }
  int *sendindex_global = new int[senddispl_global[numnode]];
  int *recvindex_global = new int[recvdispl_global[numnode]];
  int *indextemp_node = new int[numcol];
  for(int node = 0; node < numnode; node++){
    int count = 0;
    int proc = node*numproc_node+myid_node;
    for(int m = coldispl[proc]; m < coldispl[proc+1]; m++)
      if(footprint_node[m]){
        int index = recvdispl_global[node] + count;
        recvindex_global[index] = m;
        indextemp_node[m] = index;
        count++;
      }
  }
  MPI_Alltoallv(recvindex_global,recvcount_global,recvdispl_global,MPI_INT,sendindex_global,sendcount_global,senddispl_global,MPI_INT,MPI_COMM_NODE_T);
  for(int m = 0; m < senddispl_global[numnode]; m++)
    sendindex_global[m] -= coldispl[myid];
  delete[] recvindex_global;
  delete[] footprint_node;
  //FIND NODE-LEVEL COMMUNICATIONS
  bool *footprint_socket = new bool[numcol];
  MPI_Allreduce(footprint,footprint_socket,numcol,MPI_C_BOOL,MPI_LOR,MPI_COMM_SOCKET);
  int sendcount_node[numsocket];
  int recvcount_node[numsocket];
  for(int socket = 0; socket < numsocket; socket++){
    int count = 0;
    for(int node = 0; node < numnode; node++){
      int proc = node*numproc_node+socket*numproc_socket+myid_socket;
      for(int m = coldispl[proc]; m < coldispl[proc+1]; m++)
        if(footprint_socket[m])
          count++;
    }
    recvcount_node[socket] = count;
  }
  MPI_Alltoall(recvcount_node,1,MPI_INT,sendcount_node,1,MPI_INT,MPI_COMM_SOCKET_T);
  int senddispl_node[numsocket+1];
  int recvdispl_node[numsocket+1];
  senddispl_node[0] = 0;
  recvdispl_node[0] = 0;
  for(int socket = 0; socket < numsocket; socket++){
    senddispl_node[socket+1] = senddispl_node[socket] + sendcount_node[socket];
    recvdispl_node[socket+1] = recvdispl_node[socket] + recvcount_node[socket];
  }
  int commap_node[numproc*numproc];
  for(int m = 0; m < numproc*numproc; m++)
    commap_node[m] = 0;
  for(int socket = 0; socket < numsocket; socket++){
    int proc = mynode*numproc_node+socket*numproc_socket+myid_socket;
    if(recvcount_node[socket])
      commap_node[myid*numproc+proc] = recvcount_node[socket];
  }
  MPI_Allreduce(MPI_IN_PLACE,commap_node,numproc*numproc,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  int sendall_node[numproc];
  int recvall_node[numproc];
  MPI_Allgather(&senddispl_node[numsocket],1,MPI_INT,sendall_node,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&recvdispl_node[numsocket],1,MPI_INT,recvall_node,1,MPI_INT,MPI_COMM_WORLD);
  if(myid == PRINTID){
    printf("\n");
    printf("INTER-SOCKET (INTRA-NODE) COMMUNICATIONS\n");
    int sendtot = 0;
    int recvtot = 0;
    for(int proc = 0; proc < numproc; proc++){
      printf("PROCESS %d SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n",proc,sendall_node[proc],numrhs,sendall_node[proc]*sizeof(double)/1.e9*numrhs,recvall_node[proc],numrhs,recvall_node[proc]*sizeof(double)/1.e9*numrhs);
      sendtot += sendall_node[proc];
      recvtot += recvall_node[proc];
    }
    printf("TOTAL COMM SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n",sendtot,numrhs,sendtot*sizeof(double)/1.e9*numrhs,recvtot,numrhs,recvtot*sizeof(double)/1.e9*numrhs);
    printf("COMM MAP:\n");
    for(int m = 0; m < numproc; m++){
      for(int n = 0; n < numproc; n++)
        printf("%d ",commap_node[m*numproc+n]);
      printf("\n");
    }
  }
  int *recvindex_node = new int[recvdispl_node[numsocket]];
  int *sendindex_node = new int[senddispl_node[numsocket]];
  int *indextemp_socket = new int[numcol];
  for(int socket = 0; socket < numsocket; socket++){
    int count = 0;
    for(int node = 0; node < numnode; node++){
      int proc = node*numproc_node+socket*numproc_socket+myid_socket;
      for(int m = coldispl[proc]; m < coldispl[proc+1]; m++)
        if(footprint_socket[m]){
          int index = recvdispl_node[socket] + count;
          recvindex_node[index] = m;
          indextemp_socket[m] = index;
          count++;
        }
    }
  }
  MPI_Alltoallv(recvindex_node,recvcount_node,recvdispl_node,MPI_INT,sendindex_node,sendcount_node,senddispl_node,MPI_INT,MPI_COMM_SOCKET_T);
  for(int m = 0; m < senddispl_node[numsocket]; m++)
    sendindex_node[m] = indextemp_node[sendindex_node[m]];
  delete[] footprint_socket;
  delete[] recvindex_node;
  delete[] indextemp_node;
  //FIND SOCKET-LEVEL COMMUNICATIONS
  int sendcount_socket[numproc_socket];
  int recvcount_socket[numproc_socket];
  for(int proc_socket = 0; proc_socket < numproc_socket; proc_socket++){
    int count = 0;
    for(int socket = 0; socket < numsocket; socket++)
      for(int node = 0; node < numnode; node++){
        int proc = node*numproc_node+socket*numproc_socket+proc_socket;
        for(int m = coldispl[proc]; m < coldispl[proc+1]; m++)
          if(footprint[m])
            count++;
      }
    recvcount_socket[proc_socket] = count;
  }
  MPI_Alltoall(recvcount_socket,1,MPI_INT,sendcount_socket,1,MPI_INT,MPI_COMM_SOCKET);
  int senddispl_socket[numproc_socket+1];
  int recvdispl_socket[numproc_socket+1];
  senddispl_socket[0] = 0;
  recvdispl_socket[0] = 0;
  for(int proc_socket = 0; proc_socket < numproc_socket; proc_socket++){
    senddispl_socket[proc_socket+1] = senddispl_socket[proc_socket] + sendcount_socket[proc_socket];
    recvdispl_socket[proc_socket+1] = recvdispl_socket[proc_socket] + recvcount_socket[proc_socket];
  }
  int sendall_socket[numproc];
  int recvall_socket[numproc];
  MPI_Allgather(&senddispl_socket[numproc_socket],1,MPI_INT,sendall_socket,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&recvdispl_socket[numproc_socket],1,MPI_INT,recvall_socket,1,MPI_INT,MPI_COMM_WORLD);
  int commap_socket[numproc*numproc];
  for(int m = 0; m < numproc*numproc; m++)
    commap_socket[m] = 0;
  for(int proc_socket = 0; proc_socket < numproc_socket; proc_socket++){
    int proc = mynode*numproc_node+mysocket*numproc_socket+proc_socket;
    if(recvcount_socket[proc_socket])
      commap_socket[myid*numproc+proc] = recvcount_socket[proc_socket];
  }
  MPI_Allreduce(MPI_IN_PLACE,commap_socket,numproc*numproc,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if(myid == PRINTID){
    printf("\n");
    printf("INTRA-SOCKET COMMUNICATIONS\n");
    int sendtot = 0;
    int recvtot = 0;
    for(int proc = 0; proc < numproc; proc++){
      printf("PROCESS %d SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n",proc,sendall_socket[proc],numrhs,sendall_socket[proc]*sizeof(double)/1.e9*numrhs,recvall_socket[proc],numrhs,recvall_socket[proc]*sizeof(double)/1.e9*numrhs);
      sendtot += sendall_socket[proc];
      recvtot += recvall_socket[proc];
    }
    printf("TOTAL COMM SEND %d x %d (%f GB) RECV %d x %d (%f GB)\n",sendtot,numrhs,sendtot*sizeof(double)/1.e9*numrhs,recvtot,numrhs,recvtot*sizeof(double)/1.e9*numrhs);
    printf("COMM MAP:\n");
    for(int m = 0; m < numproc; m++){
      for(int n = 0; n < numproc; n++)
        printf("%d ",commap_socket[m*numproc+n]);
      printf("\n");
    }
  }
  int *sendindex_socket = new int[senddispl_socket[numproc_socket]];
  int *recvindex_socket = new int[recvdispl_socket[numproc_socket]];
  int *indextemp = new int[numcol];
  for(int proc_socket = 0; proc_socket < numproc_socket; proc_socket++){
    int count = 0;
    for(int socket = 0; socket < numsocket; socket++)
      for(int node = 0; node < numnode; node++){
        int proc = node*numproc_node+socket*numproc_socket+proc_socket;
        for(int m = coldispl[proc]; m < coldispl[proc+1]; m++)
          if(footprint[m]){
            int index = recvdispl_socket[proc_socket] + count;
            recvindex_socket[index] = m;
            indextemp[m] = index;
            count++;
          }
      }
  }
  MPI_Alltoallv(recvindex_socket,recvcount_socket,recvdispl_socket,MPI_INT,sendindex_socket,sendcount_socket,senddispl_socket,MPI_INT,MPI_COMM_SOCKET);
  for(int m = 0; m < senddispl_socket[numproc_socket]; m++)
    sendindex_socket[m] = indextemp_socket[sendindex_socket[m]];
  delete[] recvindex_socket;
  delete[] indextemp_socket;
  //FIND SELF COMMUNICATIONS
  int recvcount[numproc];
  for(int proc = 0; proc < numproc; proc++){
    int count = 0;
    for(int m = coldispl[proc]; m < coldispl[proc+1]; m++)
      if(footprint[m])
        count++;
    recvcount[proc] = count;
  }
  int recvdispl[numproc+1];
  recvdispl[0] = 0;
  for(int proc = 0; proc < numproc; proc++)
    recvdispl[proc+1] = recvdispl[proc] + recvcount[proc];
  int *recvindex = new int[recvdispl[numproc]];
  for(int proc = 0; proc < numproc; proc++){
    int count = 0;
    for(int m = coldispl[proc]; m < coldispl[proc+1]; m++)
      if(footprint[m]){
        int index = recvdispl[proc] + count;
        recvindex[index] = indextemp[m];
	indextemp[m] = index;
        count++;
      }
  }
  delete[] footprint;
  //UPDATE CSR INDICES
  for(long n = 0; n < csrdispl[mynumrow]; n++)
    csrindex[n] = indextemp[csrindex[n]];
  delete[] indextemp;

  //CONSTRUCT MULTIPLE RHS PACKING & UNPACKING INDICES
  int *tempindex_global = sendindex_global;
  sendindex_global = new int[senddispl_global[numnode]*numrhs];
  int *unpack_global = new int[recvdispl_global[numnode]*numrhs];
  for(int node = 0; node < numnode; node++)
    for(int k = 0; k < numrhs; k++){
      for(int m = 0; m < sendcount_global[node]; m++)
        sendindex_global[senddispl_global[node]*numrhs+sendcount_global[node]*k+m] = k*mynumcol+tempindex_global[senddispl_global[node]+m];
      for(int m = 0; m < recvcount_global[node]; m++)
        unpack_global[k*recvdispl_global[numnode]+recvdispl_global[node]+m] = recvdispl_global[node]*numrhs+recvcount_global[node]*k+m;
    }
  delete[] tempindex_global;
  int *tempindex_node = sendindex_node;
  sendindex_node = new int[senddispl_node[numsocket]*numrhs];
  int *unpack_node = new int[recvdispl_node[numsocket]*numrhs];
  for(int socket = 0; socket < numsocket; socket++)
    for(int k = 0; k < numrhs; k++){
      for(int m = 0; m < sendcount_node[socket]; m++)
        sendindex_node[senddispl_node[socket]*numrhs+sendcount_node[socket]*k+m] = unpack_global[k*recvdispl_global[numnode]+tempindex_node[senddispl_node[socket]+m]];
      for(int m = 0; m < recvcount_node[socket]; m++)
        unpack_node[k*recvdispl_node[numsocket]+recvdispl_node[socket]+m] = recvdispl_node[socket]*numrhs+recvcount_node[socket]*k+m;
    }
  delete[] unpack_global;
  delete[] tempindex_node;
  int *tempindex_socket = sendindex_socket;
  sendindex_socket = new int[senddispl_socket[numproc_socket]*numrhs];
  int *unpack_socket = new int[recvdispl_socket[numproc_socket]*numrhs];
  for(int proc_socket = 0; proc_socket < numproc_socket; proc_socket++)
    for(int k = 0; k < numrhs; k ++){
      for(int m = 0; m < sendcount_socket[proc_socket]; m++)
        sendindex_socket[senddispl_socket[proc_socket]*numrhs+sendcount_socket[proc_socket]*k+m] = unpack_node[k*recvdispl_node[numsocket]+tempindex_socket[senddispl_socket[proc_socket]+m]];
      for(int m = 0; m < recvcount_socket[proc_socket]; m++)
        unpack_socket[k*recvdispl_socket[numproc_socket]+recvdispl_socket[proc_socket]+m] = recvdispl_socket[proc_socket]*numrhs+recvcount_socket[proc_socket]*k+m;
    }
  delete[] unpack_node;
  delete[] tempindex_socket;
  int *tempindex = recvindex;
  recvindex = new int[recvdispl[numproc]*numrhs];
  for(int k = 0; k < numrhs; k++)
    for(int m = 0; m < recvdispl[numproc]; m++)
        recvindex[k*recvdispl[numproc]+m] = unpack_socket[k*recvdispl[numproc]+tempindex[m]];
  delete[] unpack_socket;
  delete[] tempindex;


  int commsendcount_global[numnode];
  int commrecvcount_global[numnode];
  int commsenddispl_global[numnode+1];
  int commrecvdispl_global[numnode+1];
  for(int node = 0; node < numnode; node++){
    commsendcount_global[node] = sendcount_global[node]*numrhs;
    commrecvcount_global[node] = recvcount_global[node]*numrhs;
    commsenddispl_global[node] = senddispl_global[node]*numrhs;
    commrecvdispl_global[node] = recvdispl_global[node]*numrhs;
  }
  commsenddispl_global[numnode] = senddispl_global[numnode]*numrhs;
  commrecvdispl_global[numnode] = recvdispl_global[numnode]*numrhs;
  int commsendcount_node[numsocket];
  int commrecvcount_node[numsocket];
  int commsenddispl_node[numsocket+1];
  int commrecvdispl_node[numsocket+1];
  for(int socket = 0; socket < numsocket; socket++){
    commsendcount_node[socket] = sendcount_node[socket]*numrhs;
    commrecvcount_node[socket] = recvcount_node[socket]*numrhs;
    commsenddispl_node[socket] = senddispl_node[socket]*numrhs;
    commrecvdispl_node[socket] = recvdispl_node[socket]*numrhs;
  }
  commsenddispl_node[numsocket] = senddispl_node[numsocket]*numrhs;
  commrecvdispl_node[numsocket] = recvdispl_node[numsocket]*numrhs;
  int commsendcount_socket[numproc_socket];
  int commrecvcount_socket[numproc_socket];
  int commsenddispl_socket[numproc_socket+1];
  int commrecvdispl_socket[numproc_socket+1];
  for(int proc_socket = 0; proc_socket < numproc_socket; proc_socket++){
    commsendcount_socket[proc_socket] = sendcount_socket[proc_socket]*numrhs;
    commrecvcount_socket[proc_socket] = recvcount_socket[proc_socket]*numrhs;
    commsenddispl_socket[proc_socket] = senddispl_socket[proc_socket]*numrhs;
    commrecvdispl_socket[proc_socket] = recvdispl_socket[proc_socket]*numrhs;
  }
  commsenddispl_socket[numproc_socket] = senddispl_socket[numproc_socket]*numrhs;
  commrecvdispl_socket[numproc_socket] = recvdispl_socket[numproc_socket]*numrhs;

  double *sendbuff_global = new double[senddispl_global[numnode]*numrhs];
  double *recvbuff_global = new double[recvdispl_global[numnode]*numrhs];
  double *sendbuff_node = new double[senddispl_node[numsocket]*numrhs];
  double *recvbuff_node = new double[recvdispl_node[numsocket]*numrhs];
  double *sendbuff_socket = new double[senddispl_socket[numproc_socket]*numrhs];
  double *recvbuff_socket = new double[recvdispl_socket[numproc_socket]*numrhs];
  double *recvbuff = new double[recvdispl[numproc]*numrhs];

  #pragma omp parallel for
  for(int m = 0; m < senddispl_global[numnode]*numrhs; m++)
    sendbuff_global[m] = B[sendindex_global[m]];
  MPI_Alltoallv(sendbuff_global,commsendcount_global,commsenddispl_global,MPI_DOUBLE,recvbuff_global,commrecvcount_global,commrecvdispl_global,MPI_DOUBLE,MPI_COMM_NODE_T);
  #pragma omp parallel for
  for(int m = 0; m < senddispl_node[numsocket]*numrhs; m++)
    sendbuff_node[m] = recvbuff_global[sendindex_node[m]];
  MPI_Alltoallv(sendbuff_node,commsendcount_node,commsenddispl_node,MPI_DOUBLE,recvbuff_node,commrecvcount_node,commrecvdispl_node,MPI_DOUBLE,MPI_COMM_SOCKET_T);
  #pragma omp parallel for
  for(int m = 0; m < senddispl_socket[numproc_socket]*numrhs; m++)
    sendbuff_socket[m] = recvbuff_node[sendindex_socket[m]];
  MPI_Alltoallv(sendbuff_socket,commsendcount_socket,commsenddispl_socket,MPI_DOUBLE,recvbuff_socket,commrecvcount_socket,commrecvdispl_socket,MPI_DOUBLE,MPI_COMM_SOCKET);
  #pragma omp parallel for
  for(int m = 0; m < recvdispl[numproc]*numrhs; m++)
    recvbuff[m] = recvbuff_socket[recvindex[m]];

  for(int k = 0; k < numrhs; k++)
    #pragma omp parallel for
    for(int m = 0; m < mynumrow; m++){
      double reduce = 0;
      for(int n = csrdispl[m]; n < csrdispl[m+1]; n++)
        reduce += recvbuff[k*recvdispl[numproc]+csrindex[n]]*csrvalue[n];
      C[k*mynumrow+m] = reduce;
    }

  double *Ball = new double[numcol*numrhs];
  double *Call = new double[numrow*numrhs];
  for(int k = 0; k < numrhs; k++){
    MPI_Allgatherv(C+k*mynumrow,mynumrow,MPI_DOUBLE,Call+k*numrow,numrows,rowdispl,MPI_DOUBLE,MPI_COMM_WORLD);
    MPI_Allgatherv(B+k*mynumcol,mynumcol,MPI_DOUBLE,Ball+k*numcol,numcols,coldispl,MPI_DOUBLE,MPI_COMM_WORLD);
  }
  if(myid == PRINTID){
    printf("\nB:\n");
    for(int m = 0; m < numcol; m++){
      for(int k = 0; k < numrhs; k++)
        printf("%.0f ",Ball[k*numcol+m]);
      printf("\n");
    }
    printf("\nC:\n");
    for(int m = 0; m < numrow; m++){
      for(int k = 0; k < numrhs; k++)
        printf("%.0f ",Call[k*numrow+m]);
      printf("\n");
    }
  }
  delete[] Ball;
  delete[] Call;

  delete[] sendbuff_global;
  delete[] recvbuff_global;
  delete[] sendbuff_node;
  delete[] recvbuff_node;
  delete[] sendbuff_socket;
  delete[] recvbuff_socket;
  delete[] recvbuff;

  delete[] sendindex_global;
  delete[] sendindex_node;
  delete[] sendindex_socket;
  delete[] recvindex;

  delete[] csrdispl;
  delete[] csrindex;
  delete[] csrvalue;
  delete[] B;
  delete[] C;

  MPI_Comm_free(&MPI_COMM_NODE);
  MPI_Comm_free(&MPI_COMM_NODE_T);
  MPI_Comm_free(&MPI_COMM_SOCKET);
  MPI_Comm_free(&MPI_COMM_SOCKET_T);

  MPI_Barrier(MPI_COMM_WORLD);
  if(myid == PRINTID)printf("\nTOTAL TIME: %e\n\n",omp_get_wtime()-timetotal);

  MPI_Finalize();

  return 0;
}
