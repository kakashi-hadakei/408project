
#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet {
    namespace op {

// This function is called by new-inl.h
// Any code you write should be executed by this function
        template<typename cpu, typename DType>
        void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x,
                     const mshadow::Tensor<cpu, 4, DType> &k) {
            /*
            Modify this function to implement the forward pass described in Chapter 16.
            The code in 16 is for a single image.
            We have added an additional dimension to the tensors to support an entire mini-batch
            The goal here is to be correct, not fast (this is the CPU implementation.)
            */


            int B = x.shape_[0];
            int M = y.shape_[1];
            int C = x.shape_[1];
            int H = x.shape_[2];
            int W = x.shape_[3];
            int K = k.shape_[3];
            int K0 = k.shape_[0];
            int K1 = k.shape_[1];
            int K2 = k.shape_[2];

            // int H_out = H – K + 1;
            // int W_out = W – K + 1;
            // printf("B: %d\n",B);
            // printf("M: %d\n",M);
            // printf("C: %d\n",C);
            // printf("H: %d\n",H);
            // printf("W: %d\n",W);
            // printf("K: %d\n",K);
            // printf("K0: %d\n",K0);
            // printf("K1: %d\n",K1);
            // printf("K2: %d\n",K2);
            for (int b = 0; b < B; ++b) {
                //CHECK_EQ(0, 1) << "Missing an ECE408 CPU implementation!";

                // ... a bunch of nested loops later...
                //y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];

                for (int m = 0; m < M; m++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            y[b][m][h][w] = 0;
                            for (int c = 0; c < C; c++) {
                                for (int p = 0; p < K; p++) {
                                    for (int q = 0; q < K; q++) {
                                        y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
                                    }
                                }
                            }
                        }
                    }
                }

            }


        }
    }
}

#endif