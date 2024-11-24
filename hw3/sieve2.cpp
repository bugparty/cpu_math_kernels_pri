#ifndef __SIEVE2_C__
#define __SIEVE2_C__
#include <vector>
#include <cassert>
using namespace std;
#include "include.h"
#ifdef __cplusplus
extern "C" {
#endif

    void calPrePrimes(vector<uint64_t> & primes,const int pid,  const uint64_t high_value,
                      const int pnum, const uint64_t n){
        const uint64_t sieve_low = 3;
        const uint64_t sieve_high = ceil(floor(sqrt(high_value))/2)*2-1;
        const uint64_t size = (sieve_high - sieve_low) / 2 + 1;//number of integers handled by this process

        //printf("prePrimes high_val:%d sieve_high:%d size:%d\n",  high_value, sieve_high,size);
        vector<bool> marked;
        marked.assign(size, 0);

        uint64_t index = 0;//index of current prime among all primes (only works for process 0)
        uint64_t prime = 3;//current prime broadcasted by process 0
        primes.push_back(prime);
        uint64_t prime_prime= prime * prime;
        do {
            uint64_t idxFst;//index of the first multiple among values handled by this process
            if (prime_prime > sieve_low)
                idxFst = (prime_prime - sieve_low) / 2; //located the relative offset of prime*prime after low_value
                // it is because the k*prime when k < prime, is already marked by others if k is a prime smaller
            else// prime^2 < low_value
            if (sieve_low % prime == 0)//low_value is mutiple of prime
                idxFst = 0;
            else {//low_value%prime is the difference of low_value and prime
                uint64_t first_val = prime * (floor((sieve_low / prime + 1) / 2) * 2 + 1);
                idxFst = (first_val - sieve_low) /
                         2;//The remainder tells us how far low_value is from being a multiple of prime
            }
            for (uint64_t i = idxFst; i < size; i += prime) {
                marked[i] = 1;
            } //mark all the mutiple of prime
            index++;

            while (marked[index] == 1)
                index++;
            prime = index * 2 + 3;//next prime is the first number not marked.
            //printf("pid:%d next prime %d\n", pid, prime);
            primes.push_back(prime);
            prime_prime = prime * prime;
        } while (prime_prime <= n);
    }
void sieve2(uint64_t *global_count, uint64_t n, int pnum, int pid) {
    const uint64_t low_value = 3 + 2 * floor(pid * ((n - 3) / 2 + 1) / pnum);//the smallest value handled by this process
    const uint64_t high_value =
            3 + 2 * floor((pid + 1) * ((n - 3) / 2 + 1) / pnum) - 2;//the largest value handled by this process
    const uint64_t size = (high_value - low_value) / 2 + 1;//number of integers handled by this process
    //printf("pid:%d low:%d high:%d size:%d\n",pid, low_value, high_value, size);

    if (1 + (n - 1) / pnum < (int) sqrt((double) n))//high_value of process 0 should be larger than floor(sqrt(n))
    {
        if (pid == 0)
            printf("Error: Too many processes.\n");
        MPI_Finalize();
        exit(0);
    }
    vector<bool> marked;
    marked.assign(size, 0);
    vector<uint64_t> primes;
    calPrePrimes(primes, pid, high_value, pnum, n);

    uint64_t index = 0;//index of current prime among all primes (in local array index)
    uint64_t index2 = 0;// index of pre Calculated sieving primes
    uint64_t prime = primes[index2];//current prime broadcasted by process 0
    //printf("pid:%d next prime: %d\n", pid, prime);
        do {
        uint64_t idxFst;//index of the first multiple among values handled by this process
        if ( prime * prime > low_value)
            idxFst = ( prime * prime - low_value) / 2; //located the relative offset of prime*prime after low_value
            // it is because the k*prime when k < prime, is already marked by others if k is a prime smaller
        else// prime^2 < low_value
        if (low_value % prime == 0)//low_value is mutiple of prime
            idxFst = 0;
        else {//low_value%prime is the difference of low_value and prime
            uint64_t first_val = prime * (floor((low_value / prime + 1) / 2) * 2 + 1);
            idxFst = (first_val - low_value) /
                     2;//The remainder tells us how far low_value is from being a multiple of prime
        }
        for (uint64_t i = idxFst; i < size; i += prime) {
            marked[i] = 1;
        } //mark all the mutiple of prime
        index++;
        //get next prime
        if(++index2 < primes.size()){
             prime = primes[index2];
            //printf("pid:%d next prime: %d\n", pid, prime);
        }else{
            while (marked[index]==1)
                index++;
            prime=(index)*2+low_value;//next prime is the first number not marked.
           // printf("pid:%d next prime2 %d\n", pid, prime);
        }
    } while ( prime * prime <= n);

//printf("prime:%d high:%d n:%d\n", prime, high_value, n);

    uint64_t count = 0;//local count of primes
for (uint64_t i = 0; i < size; i++)
        if (marked[i] == 0){
            count++;
        }
    if (pid == 0) {
        count++;//add prime 2

    }
    //printf("pid:%d, prime count:%d\n", pid, count);
    MPI_Reduce(&count, global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
}
    #ifdef __cplusplus
    }
    #endif
#endif