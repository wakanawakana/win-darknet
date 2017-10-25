#include "avgpool_layer.h"
#include "cuda.h"
#include <stdio.h>

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c, int size, int stride, int padding)
{
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + 2*padding)/stride;
    l.out_h = (h + 2*padding)/stride;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    #ifdef GPU
	if (gpu_index >= 0){
		l.forward_gpu = forward_avgpool_layer_gpu;
		l.backward_gpu = backward_avgpool_layer_gpu;
		l.output_gpu = cuda_make_array(l.output, output_size);
		l.delta_gpu = cuda_make_array(l.delta, output_size);
	}
    #endif
	fprintf(stderr, "avg          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
	return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;

    l->out_w = (w + 2*l->pad)/l->stride;
    l->out_h = (h + 2*l->pad)/l->stride;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

//    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
	if (gpu_index >= 0){
		cuda_free((float *)l->indexes_gpu);
		cuda_free(l->output_gpu);
		cuda_free(l->delta_gpu);
//		l->indexes_gpu = cuda_make_int_array(output_size);
		l->output_gpu = cuda_make_array(l->output, output_size);
		l->delta_gpu = cuda_make_array(l->delta, output_size);
	}
    #endif
}

void forward_avgpool_layer(const avgpool_layer l, network_state state)
{
	int b, i, j, k, m, n;
	int w_offset = -l.pad;
	int h_offset = -l.pad;

	int h = l.out_h;
	int w = l.out_w;
	int c = l.c;

	float mm = 1.0f / (l.size * l.size);
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
			for (i = 0; i < h; ++i){
				for (j = 0; j < w; ++j){
					int out_index = j + w*(i + h*(k + c*b));
					float avg = 0;
					for (n = 0; n < l.size; ++n){
						for (m = 0; m < l.size; ++m){
							int cur_h = h_offset + i*l.stride + n;
							int cur_w = w_offset + j*l.stride + m;
							int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
							int valid = (cur_h >= 0 && cur_h < l.h &&
								cur_w >= 0 && cur_w < l.w);
							avg += (valid != 0) ? state.input[index] : 0;
						}
					}
					l.output[out_index] = avg * mm;
				}
            }
        }
    }
}

void backward_avgpool_layer(const avgpool_layer l, network_state state)
{
	int b, i, j, k, m, n;
	int w_offset = -l.pad;
	int h_offset = -l.pad;

	int h = l.out_h;
	int w = l.out_w;
	int c = l.c;

	float mm = 1.0f / (l.size * l.size);
	for (b = 0; b < l.batch; ++b){
		for (k = 0; k < l.c; ++k){
			for (i = 0; i < h; ++i){
				for (j = 0; j < w; ++j){
					int in_index = j + w*(i + h*(k + c*b));
					for (n = 0; n < l.size; ++n){
						for (m = 0; m < l.size; ++m){
							int cur_h = h_offset + i*l.stride + n;
							int cur_w = w_offset + j*l.stride + m;
							int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
							int valid = (cur_h >= 0 && cur_h < l.h &&
								cur_w >= 0 && cur_w < l.w);
							state.delta[index] += (valid != 0) ? l.delta[in_index] * mm : 0;
						}
					}
				}
			}
		}
	}
}

