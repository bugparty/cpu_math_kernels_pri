#ifndef __SIEVE1_C__
#define __SIEVE1_C__

#include "include.h"
/*
n: the size
pnum: processor count
*/
void sieve1(uint64_t *global_count,uint64_t n,int pnum,int pid)
{
    uint64_t low_value=3+2*floor(pid*((n-3)/2+1)/pnum);//the smallest value handled by this process
    uint64_t high_value=3+2*floor((pid+1)*((n-3)/2+1)/pnum)-2;//the largest value handled by this process
    uint64_t size=(high_value-low_value)/2+1;//number of integers handled by this process

    if (1+(n-1)/pnum<(int)sqrt((double)n))//high_value of process 0 should be larger than floor(sqrt(n))
    {
        if (pid==0)
            printf("Error: Too many processes.\n");
        MPI_Finalize();
        exit(0);
    }

    char *marked=(char*)malloc(size);//array for marking multiples. 1 means multiple and 0 means prime
    if (marked==NULL)
    {
        printf("Error: Cannot allocate enough memory.\n");
        MPI_Finalize();
        exit(0);
    }
    memset(marked,0,size);

    uint64_t index=0;//index of current prime among all primes (only works for process 0)
    uint64_t prime=3;//current prime broadcasted by process 0
    do {
        uint64_t idxFst;//index of the first multiple among values handled by this process
        if (prime * prime > low_value)
            idxFst = (prime * prime - low_value) / 2; //located the relative offset of prime*prime after low_value
            // it is because the k*prime when k < prime, is already marked by others if k is a prime smaller 
        else// prime^2 < low_value
        if (low_value % prime == 0)//low_value is mutiple of prime
            idxFst = 0;
        else {//low_value%prime is the difference of low_value and prime
            uint64_t first_val = prime*(floor((low_value/prime+1)/2)*2+1);
            idxFst = (first_val - low_value) / 2;//The remainder tells us how far low_value is from being a multiple of prime
        }
        for (uint64_t i=idxFst;i<size;i+=prime){
            marked[i]=1;
        } //mark all the mutiple of prime
            
        index++;
        if (pid==0)// process 0
        {
            while (marked[index]==1)
                index++;
            prime=index*2+3;//next prime is the first number not marked.
            //printf("next prime %d\n", prime);
        }
        MPI_Bcast(&prime,1,MPI_INT,0,MPI_COMM_WORLD);//void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm
    } while (prime*prime<=n);
    //printf("prime:%d high:%d n:%d\n", prime, high_value, n);
    
    unsigned long long count=0;//local count of primes
    for (unsigned long long i=0;i<size;i++)
        if (marked[i]==0)
            count++;
    if(pid ==0){
        count++;//add prime 2
    }
    //printf("pid:%d, prime count:%d\n", pid, count);
    MPI_Reduce(&count,global_count,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
}

#endif