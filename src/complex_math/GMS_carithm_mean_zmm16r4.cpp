




/*MIT License
Copyright (c) 2020 Bernard Gingold
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <immintrin.h>
#include "GMS_carithm_mean_zmm16r4.h"



                   
                   void gms::math::cmean_arithm_u10x_zmm16r4_u(const float * __restrict xre,
                                                               const float * __restrict xim,
                                                               float * __restrict mre,
                                                               float * __restrict mim,
                                                               const std::size_t n) {

                        if(__builtin_expect(0ull==n,0)) {return;}
                          __m512 zmm0,zmm1,zmm2,zmm3;
                          __m512 zmm4,zmm5,zmm6,zmm7;
                          __m512 zmm8,zmm9,zmm10,zmm11;
                          __m512 zmm12,zmm13,zmm14,zmm15;
                          __m512 zmm16,zmm17,zmm18,zmm19;
                          __m512 redr[10] = {_mm512_setzero_ps()};
                          __m512 redi[10] = {_mm512_setzero_ps()};
                          float re,im;
                          const float invN = 1.0f/static_cast<float>(n);
                          std::size_t i; 
                          re = 0.0f;
                          im = 0.0f;  
                         for(i = 0ull; (i+159ull) < n; i += 160ull) {
#if (CARITHM_MEAN_ZMM16R4_SOFT_PREFETCH) == 1
                            _mm_prefetch((const char*)&xre[i+0ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+0ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xre[i+16ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+16ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xre[i+32ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+32ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xre[i+48ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+48ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xre[i+64ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+64ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xre[i+80ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+80ull],_MM_HINT_T0);  
                            _mm_prefetch((const char*)&xre[i+96ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+96ull],_MM_HINT_T0); 
                            _mm_prefetch((const char*)&xre[i+112ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+112ull],_MM_HINT_T0); 
                            _mm_prefetch((const char*)&xre[i+128ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+128ull],_MM_HINT_T0); 
                            _mm_prefetch((const char*)&xre[i+144ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+144ull],_MM_HINT_T0); 
#endif 
                            zmm0    = _mm512_loadu_ps(&xre[i+0ull]);
                            zmm1    = _mm512_loadu_ps(&xim[i+0ull]);
                            redr[0] = _mm512_add_ps(redr[0],zmm0);
                            redi[0] = _mm512_add_ps(redi[0],zmm1);
                            zmm2    = _mm512_loadu_ps(&xre[i+16ull]);
                            zmm3    = _mm512_loadu_ps(&xim[i+16ull]);
                            redr[1] = _mm512_add_ps(redr[1],zmm2);
                            redi[1] = _mm512_add_ps(redi[1],zmm3);
                            zmm4    = _mm512_loadu_ps(&xre[i+32ull]);
                            zmm5    = _mm512_loadu_ps(&xim[i+32ull]);
                            redr[2] = _mm512_add_ps(redr[2],zmm4);
                            redi[2] = _mm512_add_ps(redi[2],zmm5);
                            zmm6    = _mm512_loadu_ps(&xre[i+48ull]);
                            zmm7    = _mm512_loadu_ps(&xim[i+48ull]);
                            redr[3] = _mm512_add_ps(redr[3],zmm6);
                            redi[3] = _mm512_add_ps(redi[3],zmm7);
                            zmm8    = _mm512_loadu_ps(&xre[i+64ull]);
                            zmm9    = _mm512_loadu_ps(&xim[i+64ull]);
                            redr[4] = _mm512_add_ps(redr[4],zmm8);
                            redi[4] = _mm512_add_ps(redi[4],zmm9);
                            zmm10   = _mm512_loadu_ps(&xre[i+80ull]);
                            zmm11   = _mm512_loadu_ps(&xim[i+80ull]);
                            redr[5] = _mm512_add_ps(redr[5],zmm10);
                            redi[5] = _mm512_add_ps(redi[5],zmm11);
                            zmm12   = _mm512_loadu_ps(&xre[i+96ull]);
                            zmm13   = _mm512_loadu_ps(&xim[i+96ull]);
                            redr[6] = _mm512_add_ps(redr[6],zmm12);
                            redi[6] = _mm512_add_ps(redi[6],zmm13);
                            zmm14   = _mm512_loadu_ps(&xre[i+112ull]);
                            zmm15   = _mm512_loadu_ps(&xim[i+112ull]);
                            redr[7] = _mm512_add_ps(redr[7],zmm14);
                            redi[7] = _mm512_add_ps(redi[7],zmm15);
                            
                            zmm16   = _mm512_loadu_ps(&xre[i+128ull]);
                            zmm17   = _mm512_loadu_ps(&xim[i+128ull]);
                            redr[8] = _mm512_add_ps(redr[8],zmm16);
                            redi[8] = _mm512_add_ps(redi[8],zmm17);
                            zmm18   = _mm512_loadu_ps(&xre[i+144ull]);
                            zmm19   = _mm512_loadu_ps(&xim[i+144ull]);
                            redr[9] = _mm512_add_ps(redr[9],zmm18);
                            redi[9] = _mm512_add_ps(redi[9],zmm19);
                       }

                             redr[0] = _mm512_add_ps(redr[0],redr[5]);
                             redi[0] = _mm512_add_ps(redi[0],redi[5]);
                             redr[1] = _mm512_add_ps(redr[1],redr[6]);
                             redi[1] = _mm512_add_ps(redi[1],redi[6]);
                             redr[2] = _mm512_add_ps(redr[2],redr[7]);
                             redi[2] = _mm512_add_ps(redi[2],redi[7]);
                             redr[3] = _mm512_add_ps(redr[3],redr[8]);
                             redi[3] = _mm512_add_ps(redi[3],redi[8]);
                             redr[4] = _mm512_add_ps(redr[4],redr[9]);
                             redi[4] = _mm512_add_ps(redi[4],redi[9]);

                        for(; (i+79ull) < n; i += 80ull) {
                            zmm0    = _mm512_loadu_ps(&xre[i+0ull]);
                            zmm1    = _mm512_loadu_ps(&xim[i+0ull]);
                            redr[0] = _mm512_add_ps(redr[0],zmm0);
                            redi[0] = _mm512_add_ps(redi[0],zmm1);
                            zmm2    = _mm512_loadu_ps(&xre[i+16ull]);
                            zmm3    = _mm512_loadu_ps(&xim[i+16ull]);
                            redr[1] = _mm512_add_ps(redr[1],zmm2);
                            redi[1] = _mm512_add_ps(redi[1],zmm3);
                            zmm4    = _mm512_loadu_ps(&xre[i+32ull]);
                            zmm5    = _mm512_loadu_ps(&xim[i+32ull]);
                            redr[2] = _mm512_add_ps(redr[2],zmm4);
                            redi[2] = _mm512_add_ps(redi[2],zmm5);
                            zmm6    = _mm512_loadu_ps(&xre[i+48ull]);
                            zmm7    = _mm512_loadu_ps(&xim[i+48ull]);
                            redr[3] = _mm512_add_ps(redr[3],zmm6);
                            redi[3] = _mm512_add_ps(redi[3],zmm7);
                            zmm8    = _mm512_loadu_ps(&xre[i+64ull]);
                            zmm9    = _mm512_loadu_ps(&xim[i+64ull]);
                            redr[4] = _mm512_add_ps(redr[4],zmm8);
                            redi[4] = _mm512_add_ps(redi[4],zmm9); 
                      }

                             redr[0] = _mm512_add_ps(redr[0],redr[2]);
                             redi[0] = _mm512_add_ps(redi[0],redi[2]);
                             redr[1] = _mm512_add_ps(redr[1],redr[3]);
                             redi[1] = _mm512_add_ps(redi[1],redi[3]);
                             redr[0] = _mm512_add_ps(redr[0],redr[4]);
                             redi[0] = _mm512_add_ps(redi[0],redi[4]);

                       for(; (i+31ull) < n; i += 32ull) {
                            zmm0    = _mm512_loadu_ps(&xre[i+0ull]);
                            zmm1    = _mm512_loadu_ps(&xim[i+0ull]);
                            redr[0] = _mm512_add_ps(redr[0],zmm0);
                            redi[0] = _mm512_add_ps(redi[0],zmm1);
                            zmm2    = _mm512_loadu_ps(&xre[i+16ull]);
                            zmm3    = _mm512_loadu_ps(&xim[i+16ull]);
                            redr[1] = _mm512_add_ps(redr[1],zmm2);
                            redi[1] = _mm512_add_ps(redi[1],zmm3); 
                      }

                              redr[0] = _mm512_add_ps(redr[0],redr[1]);
                              redi[0] = _mm512_add_ps(redi[0],redi[1]); 

                      for(; (i+15ull) < n; i += 16ull) {
                            zmm0    = _mm512_loadu_ps(&xre[i+0]);
                            zmm1    = _mm512_loadu_ps(&xim[i+0]);
                            redr[0] = _mm512_add_ps(redr[0],zmm0);
                            redi[0] = _mm512_add_ps(redi[0],zmm1);
                     } 

                      for(; (i+0) < n; i += 1) {
                            const float xr = xre[i];
                            const float xi = xim[i];
                            re += xr;
                            im += xi;
                     }

                      re += _mm512_reduce_add_ps(redr[0]);
                      *mre = re*invN;
                      im += _mm512_reduce_add_ps(redi[0]);
                      *mim = im*invN;
              }   


                   
                   void gms::math::cmean_arithm_u10x_zmm16r4_a(const float * __restrict __ATTR_ALIGN__(64) xre,
                                                               const float * __restrict __ATTR_ALIGN__(64) xim,
                                                               float * __restrict mre,
                                                               float * __restrict mim,
                                                               const std::size_t n) {

                        if(__builtin_expect(0ull==n,0)) {return;}
                           __m512 zmm0,zmm1,zmm2,zmm3;
                           __m512 zmm4,zmm5,zmm6,zmm7;
                           __m512 zmm8,zmm9,zmm10,zmm11;
                           __m512 zmm12,zmm13,zmm14,zmm15;
                           __m512 zmm16,zmm17,zmm18,zmm19;
                           __ATTR_ALIGN__(64)  __m512 redr[10] = {_mm512_setzero_ps()};
                           __ATTR_ALIGN__(64)  __m512 redi[10] = {_mm512_setzero_ps()};
                           float re,im;
                           const float invN = 1.0f/static_cast<float>(n);
                           std::size_t  i; 
                           re = 0.0f;
                           im = 0.0f;  
                        for(i = 0; (i+159ull) < n; i += 160ull) {
#if (CARITHM_MEAN_ZMM16R4_SOFT_PREFETCH) == 1
                            _mm_prefetch((const char*)&xre[i+0ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+0ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xre[i+16ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+16ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xre[i+32ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+32ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xre[i+48ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+48ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xre[i+64ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+64ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xre[i+80ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+80ull],_MM_HINT_T0);  
                            _mm_prefetch((const char*)&xre[i+96ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+96ull],_MM_HINT_T0); 
                            _mm_prefetch((const char*)&xre[i+112ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+112ull],_MM_HINT_T0); 
                            _mm_prefetch((const char*)&xre[i+128ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+128ull],_MM_HINT_T0); 
                            _mm_prefetch((const char*)&xre[i+144ull],_MM_HINT_T0);
                            _mm_prefetch((const char*)&xim[i+144ull],_MM_HINT_T0); 
#endif                             
                            zmm0    = _mm512_load_ps(&xre[i+0ull]);
                            zmm1    = _mm512_load_ps(&xim[i+0ull]);
                            redr[0] = _mm512_add_ps(redr[0],zmm0);
                            redi[0] = _mm512_add_ps(redi[0],zmm1);
                            zmm2    = _mm512_load_ps(&xre[i+16ull]);
                            zmm3    = _mm512_load_ps(&xim[i+16ull]);
                            redr[1] = _mm512_add_ps(redr[1],zmm2);
                            redi[1] = _mm512_add_ps(redi[1],zmm3);
                            zmm4    = _mm512_load_ps(&xre[i+32ull]);
                            zmm5    = _mm512_load_ps(&xim[i+32ull]);
                            redr[2] = _mm512_add_ps(redr[2],zmm4);
                            redi[2] = _mm512_add_ps(redi[2],zmm5);
                            zmm6    = _mm512_load_ps(&xre[i+48ull]);
                            zmm7    = _mm512_load_ps(&xim[i+48ull]);
                            redr[3] = _mm512_add_ps(redr[3],zmm6);
                            redi[3] = _mm512_add_ps(redi[3],zmm7);
                            zmm8    = _mm512_load_ps(&xre[i+64ull]);
                            zmm9    = _mm512_load_ps(&xim[i+64ull]);
                            redr[4] = _mm512_add_ps(redr[4],zmm8);
                            redi[4] = _mm512_add_ps(redi[4],zmm9);
                            zmm10   = _mm512_load_ps(&xre[i+80ull]);
                            zmm11   = _mm512_load_ps(&xim[i+80ull]);
                            redr[5] = _mm512_add_ps(redr[5],zmm10);
                            redi[5] = _mm512_add_ps(redi[5],zmm11);
                            zmm12   = _mm512_load_ps(&xre[i+96ull]);
                            zmm13   = _mm512_load_ps(&xim[i+96ull]);
                            redr[6] = _mm512_add_ps(redr[6],zmm12);
                            redi[6] = _mm512_add_ps(redi[6],zmm13);
                            zmm14   = _mm512_load_ps(&xre[i+112ull]);
                            zmm15   = _mm512_load_ps(&xim[i+112ull]);
                            redr[7] = _mm512_add_ps(redr[7],zmm14);
                            redi[7] = _mm512_add_ps(redi[7],zmm15);
                            zmm16   = _mm512_load_ps(&xre[i+128ull]);
                            zmm17   = _mm512_load_ps(&xim[i+128ull]);
                            redr[8] = _mm512_add_ps(redr[8],zmm16);
                            redi[8] = _mm512_add_ps(redi[8],zmm17);
                            zmm18   = _mm512_load_ps(&xre[i+144ull]);
                            zmm19   = _mm512_load_ps(&xim[i+144ull]);
                            redr[9] = _mm512_add_ps(redr[9],zmm18);
                            redi[9] = _mm512_add_ps(redi[9],zmm19);
                       }

                             redr[0] = _mm512_add_ps(redr[0],redr[5]);
                             redi[0] = _mm512_add_ps(redi[0],redi[5]);
                             redr[1] = _mm512_add_ps(redr[1],redr[6]);
                             redi[1] = _mm512_add_ps(redi[1],redi[6]);
                             redr[2] = _mm512_add_ps(redr[2],redr[7]);
                             redi[2] = _mm512_add_ps(redi[2],redi[7]);
                             redr[3] = _mm512_add_ps(redr[3],redr[8]);
                             redi[3] = _mm512_add_ps(redi[3],redi[8]);
                             redr[4] = _mm512_add_ps(redr[4],redr[9]);
                             redi[4] = _mm512_add_ps(redi[4],redi[9]);

                        for(; (i+79ull) < n; i += 80ull) {
                            zmm0    = _mm512_load_ps(&xre[i+0ull]);
                            zmm1    = _mm512_load_ps(&xim[i+0ull]);
                            redr[0] = _mm512_add_ps(redr[0],zmm0);
                            redi[0] = _mm512_add_ps(redi[0],zmm1);
                            zmm2    = _mm512_load_ps(&xre[i+16ull]);
                            zmm3    = _mm512_load_ps(&xim[i+16ull]);
                            redr[1] = _mm512_add_ps(redr[1],zmm2);
                            redi[1] = _mm512_add_ps(redi[1],zmm3);
                            zmm4    = _mm512_load_ps(&xre[i+32ull]);
                            zmm5    = _mm512_load_ps(&xim[i+32ull]);
                            redr[2] = _mm512_add_ps(redr[2],zmm4);
                            redi[2] = _mm512_add_ps(redi[2],zmm5);
                            zmm6    = _mm512_load_ps(&xre[i+48ull]);
                            zmm7    = _mm512_load_ps(&xim[i+48ull]);
                            redr[3] = _mm512_add_ps(redr[3],zmm6);
                            redi[3] = _mm512_add_ps(redi[3],zmm7);
                            zmm8    = _mm512_load_ps(&xre[i+64ull]);
                            zmm9    = _mm512_load_ps(&xim[i+64ull]);
                            redr[4] = _mm512_add_ps(redr[4],zmm8);
                            redi[4] = _mm512_add_ps(redi[4],zmm9); 
                      }

                             redr[0] = _mm512_add_ps(redr[0],redr[2]);
                             redi[0] = _mm512_add_ps(redi[0],redi[2]);
                             redr[1] = _mm512_add_ps(redr[1],redr[3]);
                             redi[1] = _mm512_add_ps(redi[1],redi[3]);
                             redr[0] = _mm512_add_ps(redr[0],redr[4]);
                             redi[0] = _mm512_add_ps(redi[0],redi[4]);

                       for(; (i+31ull) < n; i += 32ull) {
                            zmm0    = _mm512_load_ps(&xre[i+0ull]);
                            zmm1    = _mm512_load_ps(&xim[i+0ull]);
                            redr[0] = _mm512_add_ps(redr[0],zmm0);
                            redi[0] = _mm512_add_ps(redi[0],zmm1);
                            zmm2    = _mm512_load_ps(&xre[i+16ull]);
                            zmm3    = _mm512_load_ps(&xim[i+16ull]);
                            redr[1] = _mm512_add_ps(redr[1],zmm2);
                            redi[1] = _mm512_add_ps(redi[1],zmm3); 
                      }

                              redr[0] = _mm512_add_ps(redr[0],redr[1]);
                              redi[0] = _mm512_add_ps(redi[0],redi[1]); 

                      for(; (i+15ull) < n; i += 16ull) {
                            zmm0    = _mm512_load_ps(&xre[i+0ull]);
                            zmm1    = _mm512_load_ps(&xim[i+0ull]);
                            redr[0] = _mm512_add_ps(redr[0],zmm0);
                            redi[0] = _mm512_add_ps(redi[0],zmm1);
                     } 

                      for(; (i+0ull) < n; i += 1ull) {
                            const float xr = xre[i];
                            const float xi = xim[i];
                            re += xr;
                            im += xi;
                     }

                      re += _mm512_reduce_add_ps(redr[0]);
                      *mre = re*invN;
                      im += _mm512_reduce_add_ps(redi[0]);
                      *mim = im*invN;
              }     
  


     
