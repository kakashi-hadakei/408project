
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 24
#define TILE_WIDTH_UNROLL 512 
#define TILE_WIDTH_UNROLL_B 2
#define TILE_WIDTH_CNN 32
#define MAP_SIZE 28
#define TILE_WIDTH_B 1
#define TILE_WIDTH_GLOBAL 8
#define M 50
#define C 1
#define H 28
#define W 28
#define K 5
#define H_out 24
#define W_out 24
#define WROW 50
#define WCOL 25
#define XROW 25
#define XCOL 576
#define INPUT_DIMENSION 784
#define MASK_DIMENSION 25
#define OUTPUT_DIMENSION 576
#define MINI_DIMENSION 28800
#define BATCH_SIZE 5000
#define ONE_TW 32
#define TWO_TW 64
#define THREE_TW 96
#define FOUR_TW 128
#define FIVE_TW 160
#define SIX_TW 192
#define SEVEN_TW 224
#define EIGHT_TW 256
#define NINE_TW 288
#define TEN_TW 320
#define ELEVEN_TW 352
#define TWELVE_TW 384
#define THIRTEEN_TW 416
#define FOURTEEN_TW 448
#define FIVETEEN_TW 480
#define SIXTEEN_TW 512
#define SEVENTEEN_TW 544
#define EIGHTEEN_TW 576
#define NINETEEN_TW 608
#define TWENTY_TW 640
#define TWENTY_ONE_TW 672
#define TWENTY_TWO_TW 704
#define TWENTY_THREE_TW 736
#define TWENTY_FOUR_TW 768 
#define WPT 8
#define RTS 4
#define ONE_FEATURE 14400



namespace mxnet {
    namespace op {
        __global__ void unroll_kernel(float* unrolled,const float* input,int B){
            #define input(i3,i2,i1) input[(i3) * (INPUT_DIMENSION) + (i2)*(INPUT_DIMENSION) + (i1)]
            int b = blockIdx.x;
            int elem = threadIdx.x;

            int input_row = elem / W;
            int input_col = elem % W;
            int output_srow = elem / W_out;
            int output_scol = elem % W_out;

            __shared__ float map[MAP_SIZE][MAP_SIZE];
            
            if(elem < INPUT_DIMENSION){
                map[input_row][input_col] = input(b,0,elem);
            }

            __syncthreads();

            if(elem < OUTPUT_DIMENSION){
                for(int row = output_srow;row < output_srow + K;row++){
                    for(int col = output_scol;col < output_scol + K;col++){
                        unrolled[b * ONE_FEATURE + ((row-output_srow) * K + (col-output_scol)) * OUTPUT_DIMENSION + elem] = map[row][col];
                    }
                }
            }

            #undef input
        }

        __global__ void forward_kernel_register_blocking(float* y, const float* w,const float* x,int B) {
            #define y4d(i3, i2, i1, i0) y[(i3) * (MINI_DIMENSION) + (i2)*(OUTPUT_DIMENSION) + (i1)*(W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (INPUT_DIMENSION) + (i2)*(INPUT_DIMENSION) + (i1)*(W) + i0]
            #define w4d(i3, i2, i1, i0) w[(i3) * (MASK_DIMENSION) + (i2)*(MASK_DIMENSION) + (i1)*(K) + i0]
            #define devW(i3, i2, i1, i0) devW[(i3) * (MASK_DIMENSION) + (i2)*(MASK_DIMENSION) + (i1)*(K) + i0]
            #define y3d(i2,i1,i0) y[(i2) * (MINI_DIMENSION) + (i1)*(OUTPUT_DIMENSION) + i0]
            #define w3d(i2,i1,i0) w[(i2) * (MASK_DIMENSION) + (i1)*(MASK_DIMENSION) + i0]
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int Row = blockIdx.y * TILE_WIDTH_CNN + ty;
            int Col = blockIdx.x * TILE_WIDTH_CNN + tx;
            int b = blockIdx.z;
            int m,k;
            int local_pos = tx;
            int global_pos = Col;

            __shared__ float subtileA[TILE_WIDTH_CNN][TILE_WIDTH_CNN];
            __shared__ float subtileB[TILE_WIDTH_CNN][TILE_WIDTH_CNN];
            float acc[WPT];
            
            #pragma unroll
            for (m=0; m<WPT; m++) {
                acc[m] = 0.0;
            }

            for(m = 0; m < WPT; m++){
                if(Row < WROW && (local_pos) < WCOL) {
                    subtileA[ty][local_pos] = w4d(Row,0,(local_pos) / K, (local_pos) % K);
                }
                else{
                    subtileA[ty][local_pos] = 0;
                }
                
                if(ty < XROW && (global_pos) < XCOL) {
                    //subtileB[ty][local_pos] = x4d(b,0,(global_pos) / W_out  + ty / K,(global_pos) % W_out + ty % K);
                    subtileB[ty][local_pos] = x[b * ONE_FEATURE +  ty * OUTPUT_DIMENSION + global_pos ];

                    //subtileB[ty][local_pos] = x[ty*]
                }else {
                    subtileB[ty][local_pos] = 0;
                }

                local_pos += RTS;
                global_pos += RTS;
            }   

            __syncthreads();

            #pragma unroll
            for (k=0; k<25; k++) {
                local_pos = tx;
                for (m=0; m<WPT; m++) {
                    acc[m] += subtileA[ty][k] * subtileB[k][local_pos];
                    local_pos += RTS;
                }
            }

            global_pos = Col;
            for (m=0; m<WPT; m++) {
                if (Row < WROW && global_pos < XCOL){
                    y4d(b, Row, (global_pos) / W_out, (global_pos) % W_out) = acc[m];
                }
                global_pos+= RTS;
            }   


            #undef y3d
            #undef y4d
            #undef y4d
            #undef x4d
            #undef w4d
            #undef devW
        }


        __global__ void forward_kernel_trial(float* y, const float* w,const float* x,int B) {
            #define y4d(i3, i2, i1, i0) y[(i3) * (MINI_DIMENSION) + (i2)*(OUTPUT_DIMENSION) + (i1)*(W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (INPUT_DIMENSION) + (i2)*(INPUT_DIMENSION) + (i1)*(W) + i0]
            #define w4d(i3, i2, i1, i0) w[(i3) * (MASK_DIMENSION) + (i2)*(MASK_DIMENSION) + (i1)*(K) + i0]
            #define devW(i3, i2, i1, i0) devW[(i3) * (MASK_DIMENSION) + (i2)*(MASK_DIMENSION) + (i1)*(K) + i0]
            #define y3d(i2,i1,i0) y[(i2) * (MINI_DIMENSION) + (i1)*(OUTPUT_DIMENSION) + i0]
            #define w3d(i2,i1,i0) w[(i2) * (MASK_DIMENSION) + (i1)*(MASK_DIMENSION) + i0]
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int Row = blockIdx.y * TILE_WIDTH_CNN + ty;
            int Col = blockIdx.x * TILE_WIDTH_CNN + tx;
            int b = blockIdx.z;
            int ty_tw = ty * TILE_WIDTH_CNN;

            __shared__ float subtileA[TILE_WIDTH_CNN*TILE_WIDTH_CNN];
            __shared__ float subtileB[TILE_WIDTH_CNN*TILE_WIDTH_CNN];


            float p_value;

            for(int c = Col;c < OUTPUT_DIMENSION;c+= blockDim.x * gridDim.x){

                //if(ty < XROW && c < XCOL) {
                    subtileB[ty_tw+tx] = x[b * ONE_FEATURE +  ty * OUTPUT_DIMENSION + c ];
                // }else{
                //     subtileB[ty_tw+tx] = 0;
                // }

                for(int r = Row; r < 64;r += blockDim.y * gridDim.y){
                p_value = 0;
                //if(r < WROW && tx < WCOL) {
                    subtileA[ty_tw+tx] = w3d(r,0,tx);
                // }else{
                //     subtileA[ty_tw+tx] = 0;
                // }
                __syncthreads();

                p_value += subtileA[ty_tw] * subtileB[tx];
                p_value += subtileA[ty_tw+1] * subtileB[ONE_TW+tx];
                p_value += subtileA[ty_tw+2] * subtileB[TWO_TW+tx];
                p_value += subtileA[ty_tw+3] * subtileB[THREE_TW+tx];
                p_value += subtileA[ty_tw+4] * subtileB[FOUR_TW+tx];
                p_value += subtileA[ty_tw+5] * subtileB[FIVE_TW+tx];
                p_value += subtileA[ty_tw+6] * subtileB[SIX_TW+tx];
                p_value += subtileA[ty_tw+7] * subtileB[SEVEN_TW+tx]; 
                p_value += subtileA[ty_tw+8] * subtileB[EIGHT_TW+tx];
                p_value += subtileA[ty_tw+9] * subtileB[NINE_TW+tx];
                p_value += subtileA[ty_tw+10] * subtileB[TEN_TW+tx];
                p_value += subtileA[ty_tw+11] * subtileB[ELEVEN_TW+tx];
                p_value += subtileA[ty_tw+12] * subtileB[TWELVE_TW+tx];
                p_value += subtileA[ty_tw+13] * subtileB[THIRTEEN_TW+tx];
                p_value += subtileA[ty_tw+14] * subtileB[FOURTEEN_TW+tx];
                p_value += subtileA[ty_tw+15] * subtileB[FIVETEEN_TW+tx];
                p_value += subtileA[ty_tw+16] * subtileB[SIXTEEN_TW+tx];
                p_value += subtileA[ty_tw+17] * subtileB[SEVENTEEN_TW+tx];
                p_value += subtileA[ty_tw+18] * subtileB[EIGHTEEN_TW+tx];
                p_value += subtileA[ty_tw+19] * subtileB[NINETEEN_TW+tx];
                p_value += subtileA[ty_tw+20] * subtileB[TWENTY_TW+tx];
                p_value += subtileA[ty_tw+21] * subtileB[TWENTY_ONE_TW+tx];
                p_value += subtileA[ty_tw+22] * subtileB[TWENTY_TWO_TW+tx];
                p_value += subtileA[ty_tw+23] * subtileB[TWENTY_THREE_TW+tx];
                p_value += subtileA[ty_tw+24] * subtileB[TWENTY_FOUR_TW+tx];
            
                
                if (r < WROW && c < XCOL){
                    //y4d(b, r, srow, scol) = p_value;
                    y3d(b,r,c) = p_value;
                }
                __syncthreads();
                }
            }
        }
         __global__ void forward_kernel_stable75(float* y, const float* w,const float* x,int B) {
            #define y4d(i3, i2, i1, i0) y[(i3) * (MINI_DIMENSION) + (i2)*(OUTPUT_DIMENSION) + (i1)*(W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (INPUT_DIMENSION) + (i2)*(INPUT_DIMENSION) + (i1)*(W) + i0]
            #define w4d(i3, i2, i1, i0) w[(i3) * (MASK_DIMENSION) + (i2)*(MASK_DIMENSION) + (i1)*(K) + i0]
            #define devW(i3, i2, i1, i0) devW[(i3) * (MASK_DIMENSION) + (i2)*(MASK_DIMENSION) + (i1)*(K) + i0]
            #define y3d(i2,i1,i0) y[(i2) * (MINI_DIMENSION) + (i1)*(OUTPUT_DIMENSION) + i0]
            #define w3d(i2,i1,i0) w[(i2) * (MASK_DIMENSION) + (i1)*(MASK_DIMENSION) + i0]
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int Row = blockIdx.y * TILE_WIDTH_CNN * 2 + ty;
            int Col = blockIdx.x * TILE_WIDTH_CNN + tx;
            int b = blockIdx.z;

            int srow = Col / W_out;
            int scol = Col % W_out;
            int w_loading_col = tx,x_loading_row = ty;
            int ty_tw = ty * TILE_WIDTH_CNN;

            __shared__ float subtileA[TILE_WIDTH_CNN*TILE_WIDTH_CNN];
            __shared__ float subtileB[TILE_WIDTH_CNN*TILE_WIDTH_CNN];


            float p_value;

            //if(x_loading_row < XROW && Col < XCOL) {
                subtileB[ty_tw+tx] = x4d(b,0,srow  + x_loading_row / K,scol + x_loading_row % K);
                //subtileB[ty_tw+tx] = unroll[b*ONE_FEATURE + x_loading_row * OUTPUT_DIMENSION + Col];
            // }else{
            //     subtileB[ty_tw+tx] = 0;
            // }

            for(int r = Row; r < M;r += TILE_WIDTH_CNN){
                p_value = 0;
                //if(r < WROW && w_loading_col < WCOL) {
                    //subtileA[ty_tw+tx] = w4d(r,0,w_loading_col / K,w_loading_col % K);
                    subtileA[ty_tw+tx] = w3d(r,0,w_loading_col);
                // }else{
                //     subtileA[ty_tw+tx] = 0;
                // }

                __syncthreads();

                p_value += subtileA[ty_tw] * subtileB[tx];
                p_value += subtileA[ty_tw+1] * subtileB[ONE_TW+tx];
                p_value += subtileA[ty_tw+2] * subtileB[TWO_TW+tx];
                p_value += subtileA[ty_tw+3] * subtileB[THREE_TW+tx];
                p_value += subtileA[ty_tw+4] * subtileB[FOUR_TW+tx];
                p_value += subtileA[ty_tw+5] * subtileB[FIVE_TW+tx];
                p_value += subtileA[ty_tw+6] * subtileB[SIX_TW+tx];
                p_value += subtileA[ty_tw+7] * subtileB[SEVEN_TW+tx]; 
                p_value += subtileA[ty_tw+8] * subtileB[EIGHT_TW+tx];
                p_value += subtileA[ty_tw+9] * subtileB[NINE_TW+tx];
                p_value += subtileA[ty_tw+10] * subtileB[TEN_TW+tx];
                p_value += subtileA[ty_tw+11] * subtileB[ELEVEN_TW+tx];
                p_value += subtileA[ty_tw+12] * subtileB[TWELVE_TW+tx];
                p_value += subtileA[ty_tw+13] * subtileB[THIRTEEN_TW+tx];
                p_value += subtileA[ty_tw+14] * subtileB[FOURTEEN_TW+tx];
                p_value += subtileA[ty_tw+15] * subtileB[FIVETEEN_TW+tx];
                p_value += subtileA[ty_tw+16] * subtileB[SIXTEEN_TW+tx];
                p_value += subtileA[ty_tw+17] * subtileB[SEVENTEEN_TW+tx];
                p_value += subtileA[ty_tw+18] * subtileB[EIGHTEEN_TW+tx];
                p_value += subtileA[ty_tw+19] * subtileB[NINETEEN_TW+tx];
                p_value += subtileA[ty_tw+20] * subtileB[TWENTY_TW+tx];
                p_value += subtileA[ty_tw+21] * subtileB[TWENTY_ONE_TW+tx];
                p_value += subtileA[ty_tw+22] * subtileB[TWENTY_TWO_TW+tx];
                p_value += subtileA[ty_tw+23] * subtileB[TWENTY_THREE_TW+tx];
                p_value += subtileA[ty_tw+24] * subtileB[TWENTY_FOUR_TW+tx];
            
                
                if (r < WROW && Col < XCOL){
                    //y4d(b, r, srow, scol) = p_value;
                    y3d(b,r,Col) = p_value;
                }

                __syncthreads();

            }
        }



        // This function is called by new-inl.h
        // Any code you write should be executed by this function
        template<>
        void forward(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
            int B = x.shape_[0];
            // float * devUnrollX; 
            // int sizeX = B * H_out * W_out * K * K;
            // cudaMalloc((void**)&devUnrollX,sizeof(float) * sizeX);
            // dim3 dimBlockUnroll(1024,1,1);
            // dim3 dimGridUnroll(B,1,1);
            // unroll_kernel<<<dimGridUnroll,dimBlockUnroll>>>(devUnrollX,x.dptr_,B);
            // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());


            //dim3 dimBlockCNN(RTS,TILE_WIDTH_CNN,1);
            dim3 dimBlockCNN(TILE_WIDTH_CNN,TILE_WIDTH_CNN,1);
            //dim3 dimGridCNN(2,2,B);
            //dim3 dimGridCNN(9,2,B);
            dim3 dimGridCNN(18,1,B);
            //forward_kernel_register_blocking<<<dimGridCNN,dimBlockCNN>>>(y.dptr_, w.dptr_,x.dptr_,B);
            //forward_kernel_register_blocking<<<dimGridCNN,dimBlockCNN>>>(y.dptr_, w.dptr_,devUnrollX,B);
            //forward_kernel_trial<<<dimGridCNN,dimBlockCNN>>>(y.dptr_, w.dptr_,x.dptr_,B);
            //forward_kernel_trial<<<dimGridCNN,dimBlockCNN>>>(y.dptr_, w.dptr_,devUnrollX,B);
            //forward_kernel_trial<<<dimGridCNN,dimBlockCNN>>>(y.dptr_, w.dptr_,devUnrollX,B);
            forward_kernel_stable75<<<dimGridCNN,dimBlockCNN>>>(y.dptr_, w.dptr_,x.dptr_,B);
            //forward_kernel_stable75<<<dimGridCNN,dimBlockCNN>>>(y.dptr_, w.dptr_,devUnrollX,B);
            MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
            //cudaFree(devUnrollX);
        }

        template<typename gpu, typename DType>
        void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
            assert( 0 && "No forward implementation for other datatypes needed for ECE408");
        }
    }
}

#endif
