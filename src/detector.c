#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif
static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
	list *options = read_data_cfg(datacfg);
	char *train_images = option_find_str(options, "train", "data/train.list");
	char *backup_directory = option_find_str(options, "backup", "/backup/");

	srand(time(0));
	char *base = basecfg(cfgfile);
	printf("%s\n", base);
	float avg_loss = -1;
	network *nets = calloc(ngpus, sizeof(network));

	srand(time(0));
	int seed = rand_r();
	int i;
	for (i = 0; i < ngpus; ++i){
		srand(seed);
#ifdef GPU
		if(gpu_index >= 0) cuda_set_device(gpus[i]);
#endif
		nets[i] = parse_network_cfg(cfgfile);
		if (weightfile){
			load_weights(&nets[i], weightfile);
		}
		if (clear) *nets[i].seen = 0;
		nets[i].learning_rate *= ngpus;
	}
	srand(time(0));
	network net = nets[0];

	int imgs = net.batch * net.subdivisions * ngpus;
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	data train, buffer;

	layer l = net.layers[net.n - 1];

	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_paths(train_images);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.jitter = jitter;
	args.num_boxes = l.max_boxes;
	args.d = &buffer;
	args.type = DETECTION_DATA;
	args.threads = 8;

	args.angle = net.angle;
	args.exposure = net.exposure;
	args.saturation = net.saturation;
	args.hue = net.hue;

	pthread_t load_thread = load_data(args);
	clock_t time;
	int count = 0;
	//while(i*imgs < N*120){
	while (get_current_batch(net) < net.max_batches){
		if (l.random && count++ % 10 == 0){
			printf("Resizing\n");
			int dim = (rand_r() % 10 + 10) * 32;
			if (get_current_batch(net) + 200 > net.max_batches) dim = 608;
			//int dim = (rand_r() % 4 + 16) * 32;
			printf("%d\n", dim);
			args.w = dim;
			args.h = dim;

			pthread_join(load_thread, 0);
			train = buffer;
			free_data(train);
			load_thread = load_data(args);

			for (i = 0; i < ngpus; ++i){
				resize_network(nets + i, dim, dim);
			}
			net = nets[0];
		}
		time = clock();
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data(args);

		/*
		   int k;
		   for(k = 0; k < l.max_boxes; ++k){
		   box b = float_to_box(train.y.vals[10] + 1 + k*5);
		   if(!b.x) break;
		   printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
		   }
		   image im = float_to_image(448, 448, 3, train.X.vals[10]);
		   int k;
		   for(k = 0; k < l.max_boxes; ++k){
		   box b = float_to_box(train.y.vals[10] + 1 + k*5);
		   printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
		   draw_bbox(im, b, 8, 1,0,0);
		   }
		   save_image(im, "truth11");
		   */

		printf("Loaded: %lf seconds\n", sec(clock() - time));

		time = clock();
		float loss = 0;
#ifdef GPU
		if(gpu_index >= 0){
			if(ngpus == 1){
				loss = train_network(net, train);
			} else {
				loss = train_networks(nets, ngpus, train, 4);
			}
		}
#else
		loss = train_network(net, train);
#endif
		if (avg_loss < 0) avg_loss = loss;
		avg_loss = avg_loss*.9 + loss*.1;

		i = get_current_batch(net);
		printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock() - time), i*imgs);
		if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)){
#ifdef GPU
			if(gpu_index >= 0){
				if(ngpus != 1) sync_nets(nets, ngpus, 0);
			}
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
#ifdef GPU
	if(gpu_index >= 0){
		if (ngpus != 1) sync_nets(nets, ngpus, 0);
	}
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '_');
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, box *boxes, float **probs, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, probs[i][j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_detector_detections_mod(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
	int i, j;
	float thresh = 0.01;
	for (i = 0; i < total; ++i){
		float xmin = boxes[i].x - boxes[i].w / 2.;
		float xmax = boxes[i].x + boxes[i].w / 2.;
		float ymin = boxes[i].y - boxes[i].h / 2.;
		float ymax = boxes[i].y + boxes[i].h / 2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for (j = 0; j < classes; ++j){
			if (probs[i][j] > thresh) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
				xmin, ymin, xmax, ymax);
		}
	}
}

void print_imagenet_detections(FILE *fp, int id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (probs[i][class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, probs[i][class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            get_region_boxes(l, w, h, thresh, probs, boxes, 0, map, .5);
            if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, nms);
            if (coco){
                print_cocos(fp, path, boxes, probs, l.w*l.h*l.n, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, boxes, probs, l.w*l.h*l.n, classes, w, h);
            } else {
//                print_detector_detections(fps, id, boxes, probs, l.w*l.h*l.n, classes, w, h);
				print_detector_detections_mod(fps, id, boxes, probs, l.w*l.h*l.n, classes, w, h);
			}
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_detector_recall1(char *datacfg, char *cfgfile, char *weightfile, float thresh, float iou_thresh, float nms, int stimg)
{
	list *options = read_data_cfg(datacfg);
	char *recall_images = option_find_str(options, "recall", "data/recall.list");
	network net = parse_network_cfg(cfgfile);
	layer l = net.layers[net.n - 1];
	int classes = l.classes;
	char savename[255];
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	image **alphabet = load_alphabet();

	if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

	list *plist = get_paths(recall_images);
	//    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    int j, j2, k, c;
    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
	float **probs2 = calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j)
	{
		probs[j] = calloc(classes, sizeof(float *));
		probs2[j] = calloc(classes, sizeof(float *));
		memset(probs2[j], 0, sizeof(float *));
	}


    int m = plist->size;
    int i=0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;
	float tp_iou = 0;
	int clsproposals[80];
	int clstruth[80];
	int clscorrect[80];
	float clstpiou[80];
	float clsaveiou[80];
	memset(clsproposals, 0, sizeof(clsproposals));
	memset(clstruth, 0, sizeof(clstruth));
	memset(clscorrect, 0, sizeof(clscorrect));
	memset(clstpiou, 0, sizeof(clstpiou));
	memset(clsaveiou, 0, sizeof(clsaveiou));

	for (i = 0; i < m; ++i){
		char *path = paths[i];
		image orig = load_image_color(path, 0, 0);
		image orig2 = load_image_color(path, 0, 0);
		image sized = resize_image(orig, net.w, net.h);
		char *id = basecfg(path);
		network_predict(net, sized.data);
		get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, .5);
		if (nms) do_nms(boxes, probs, l.w*l.h*l.n, classes, nms);
//		if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, nms);

		char labelpath[4096];
		find_replace(path, "images", "labels", labelpath);
		find_replace(labelpath, "JPEGImages", "labels", labelpath);
		find_replace(labelpath, ".bmp", ".txt", labelpath);
		find_replace(labelpath, ".png", ".txt", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);

		int num_labels = 0;
		int c2prop_old = 0;

		box_label *truth = read_boxes(labelpath, &num_labels);
		for (j = 0; j < num_labels; j++){
			++clstruth[truth[j].id];
		}
		c2prop_old = clsproposals[2];// cls-0 special
		for (k = 0; k < l.w*l.h*l.n; ++k){
			for (c = 0; c < classes; ++c){
				if (probs[k][c] > thresh){
					++proposals;
					++clsproposals[c];
					float best_iou = 0;
					int best_j = 0;
					for (j = 0; j < num_labels; ++j) {
						box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
						float iou = box_iou(boxes[k], t);
						if (iou > best_iou){
							best_iou = iou;
							best_j = j;
						}
					}
					fprintf(stdout, "pb:%d-%d-%d %d %f, %f, %f, %f, %f, %f\n", i, best_j, k, c, probs[k][c], best_iou, boxes[k].x, boxes[k].y, boxes[k].w, boxes[k].h);
				}
			}
			memset(probs2[k], 0.0f, sizeof(float *));
		}

		if (stimg > 0){
			draw_detections(orig, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
			sprintf(savename, "predictions/%d", i);
			save_image(orig, savename);
		}
		for (j2 = 0; j2 < num_labels; ++j2) {
			++total;
			float best_iou = 0;
			int best_k = -1;
			int best_c = 0;
			int best_j = 0;
			for (j = 0; j < num_labels; ++j) {
				box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
				for (k = 0; k < l.w*l.h*l.n; ++k){
					for (c = 0; c < classes; ++c){
						if (probs[k][c] > thresh){
							float iou = box_iou(boxes[k], t);
							//							fprintf(stdout, "%pb:d-%d-%d %d %f, %f, %f, %f, %f, %f\n", i, j, k, c, probs[k][c], iou, boxes[k].x, boxes[k].y, boxes[k].w, boxes[k].h);
							if (iou > best_iou){
								best_iou = iou;
								best_k = k;
								best_c = c;
								best_j = j;
							}
						}
					}
				}
			}

			avg_iou += best_iou;
			clsaveiou[best_c] += best_iou;

			if (best_iou > iou_thresh && best_c == truth[best_j].id){
				++correct;
				++clscorrect[best_c];

				clstpiou[best_c] += best_iou;
				tp_iou += best_iou;
			}
			if (best_k >= 0){
				//					fprintf(stdout, "   %d-%d-%d %d %f, %f, %f, %f, %f, %f\n", i, best_j, best_k, best_c, probs[best_k][best_c], best_iou, boxes[best_k].x, boxes[best_k].y, boxes[best_k].w, boxes[best_k].h);
				probs[best_k][best_c] = 0.0;
			}
		}

		if (stimg > 0){
			draw_detections(orig2, l.w*l.h*l.n, thresh, boxes, probs2, names, alphabet, l.classes);
			sprintf(savename, "predictions/%d- all and tip", i);
			save_image(orig2, savename);
		}

		if (correct > 0){
			for (j = 0; j < 3; j++){
				fprintf(stdout, "   c-%d %d %d IOU: %.2f%% TPIOU: %.2f%% Recall: %.2f%% Precision: %.2f%%\n", j, clscorrect[j], clsproposals[j], clsaveiou[j] * 100 / clstruth[j], clstpiou[j] * 100 / clscorrect[j], 100.*clscorrect[j] / clstruth[j], 100.*clscorrect[j] / clsproposals[j]);
			}
			fprintf(stdout, "   %d %d %d RPs/Img: %.2f IOU: %.2f%% TPIOU: %.2f%% Recall: %.2f%% Precision: %.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, tp_iou * 100 / correct, 100.*correct / total, 100.*correct / proposals);
		}
		else {
			fprintf(stdout, "   %d %d %d RPs/Img: %.2f IOU: %.2f%% TPIOU: %.2f%% Recall: %.2f%% Precision: %.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 0.0, 0.0, 0.0);
		}
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void validate_detector_recall2(char *datacfg, char *cfgfile, char *weightfile, float thresh, float iou_thresh, float nms, int stimg)
{
	list *options = read_data_cfg(datacfg);
	char *recall_images = option_find_str(options, "recall", "data/recall.list");
	network net = parse_network_cfg(cfgfile);
	layer l = net.layers[net.n - 1];
	int classes = l.classes;
	char savename[255];
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	image **alphabet = load_alphabet();

	if (weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	list *plist = get_paths(recall_images);
	//    list *plist = get_paths("data/voc.2007.test");
	char **paths = (char **)list_to_array(plist);

	int j, j2, k, c;
	box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
	float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
	float **probs2 = calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j)
	{
		probs[j] = calloc(classes, sizeof(float *));
		probs2[j] = calloc(classes, sizeof(float *));
		memset(probs2[j], 0, sizeof(float *));
	}


	int m = plist->size;
	int i = 0;

	int total = 0;
	int correct = 0;
	int proposals = 0;
	float avg_iou = 0;
	float tp_iou = 0;
	int clsproposals[80];
	int clstruth[80];
	int clscorrect[80];
	int cls2table[10]; // cls-0 special
	float clstpiou[80];
	float clsaveiou[80];
	memset(clsproposals, 0, sizeof(clsproposals));
	memset(clstruth, 0, sizeof(clstruth));
	memset(clscorrect, 0, sizeof(clscorrect));
	memset(clstpiou, 0, sizeof(clstpiou));
	memset(clsaveiou, 0, sizeof(clsaveiou));

	for (i = 0; i < m; ++i){
		char *path = paths[i];
		image orig = load_image_color(path, 0, 0);
		image orig2 = load_image_color(path, 0, 0);
		image sized = resize_image(orig, net.w, net.h);
		char *id = basecfg(path);
		network_predict(net, sized.data);
		get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, .5);
		if (nms) do_nms(boxes, probs, l.w*l.h*l.n, classes, nms);
		//		if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, nms);

		char labelpath[4096];
		find_replace(path, "images", "labels", labelpath);
		find_replace(labelpath, "JPEGImages", "labels", labelpath);
		find_replace(labelpath, ".bmp", ".txt", labelpath);
		find_replace(labelpath, ".png", ".txt", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);

		int num_labels = 0;
		int c2prop_old = 0;

		box_label *truth = read_boxes(labelpath, &num_labels);
		for (j = 0; j < num_labels; j++){
			++clstruth[truth[j].id];
		}
		c2prop_old = clsproposals[2];// cls-0 special
		for (k = 0; k < l.w*l.h*l.n; ++k){
			for (c = 0; c < classes; ++c){
				if (probs[k][c] > thresh){
					++proposals;
					++clsproposals[c];
					float best_iou = 0;
					int best_j = 0;
					for (j = 0; j < num_labels; ++j) {
						box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
						float iou = box_iou(boxes[k], t);
						if (iou > best_iou){
							best_iou = iou;
							best_j = j;
						}
					}
					if (c == 2){ // cls-0 special
						cls2table[clsproposals[2] - 1 - c2prop_old] = k; // cls-0 special
					}
					fprintf(stdout, "pb:%d-%d-%d %d %f, %f, %f, %f, %f, %f\n", i, best_j, k, c, probs[k][c], best_iou, boxes[k].x, boxes[k].y, boxes[k].w, boxes[k].h);
				}
			}
			memset(probs2[k], 0.0f, sizeof(float *));
		}

		clstruth[0] = clstruth[1]; // cls-0 special

		for (j2 = 0; j2 < clsproposals[2] - c2prop_old; j2++){
			float best_iou = 0;
			int best_k = -1;
			for (k = 0; k < l.w*l.h*l.n; ++k){
				if (probs[k][1] > thresh){
					float iou = box_iou(boxes[k], boxes[cls2table[j2]]);
					if (iou > best_iou && probs2[k][1] == 0.0f){
						best_k = k;
						best_iou = iou;
					}
				}
			}
			if (best_k >= 0){
				probs2[best_k][1] = probs[best_k][1];
				best_iou = 0;
				clsproposals[0]++;
				for (j = 0; j < num_labels; ++j) {
					box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
					if (probs[best_k][1] > thresh){
						float iou = box_iou(boxes[best_k], t);
						if (iou > best_iou){
							best_iou = iou;
						}
					}
				}
				clsaveiou[0] += best_iou;
				if (best_iou > iou_thresh){
					clscorrect[0]++;
					clstpiou[0] += best_iou;
				}
			}
		}

		if (stimg > 0){
			draw_detections(orig, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
			sprintf(savename, "predictions/%d", i);
			save_image(orig, savename);
		}
		for (j2 = 0; j2 < num_labels; ++j2) {
			++total;
			float best_iou = 0;
			int best_k = -1;
			int best_c = 0;
			int best_j = 0;
			for (j = 0; j < num_labels; ++j) {
				box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
				for (k = 0; k < l.w*l.h*l.n; ++k){
					for (c = 0; c < classes; ++c){
						if (probs[k][c] > thresh){
							float iou = box_iou(boxes[k], t);
							//							fprintf(stdout, "%pb:d-%d-%d %d %f, %f, %f, %f, %f, %f\n", i, j, k, c, probs[k][c], iou, boxes[k].x, boxes[k].y, boxes[k].w, boxes[k].h);
							if (iou > best_iou){
								best_iou = iou;
								best_k = k;
								best_c = c;
								best_j = j;
							}
						}
					}
				}
			}

			avg_iou += best_iou;
			clsaveiou[best_c] += best_iou;
			if (best_c == 1){// cls-0 special
				//				clsaveiou[0] += best_iou;
			}
			if (best_iou > iou_thresh && best_c == truth[best_j].id){
				++correct;
				++clscorrect[best_c];
				if (best_c == 1){// cls-0 special
					//					++clscorrect[0];
					//					clstpiou[0] += best_iou;
				}
				clstpiou[best_c] += best_iou;
				tp_iou += best_iou;
			}
			if (best_k >= 0){
				//					fprintf(stdout, "   %d-%d-%d %d %f, %f, %f, %f, %f, %f\n", i, best_j, best_k, best_c, probs[best_k][best_c], best_iou, boxes[best_k].x, boxes[best_k].y, boxes[best_k].w, boxes[best_k].h);
				probs[best_k][best_c] = 0.0;
			}
		}

		if (stimg > 0){
			draw_detections(orig2, l.w*l.h*l.n, thresh, boxes, probs2, names, alphabet, l.classes);
			sprintf(savename, "predictions/%d- all and tip", i);
			save_image(orig2, savename);
		}

		if (correct > 0){
			for (j = 0; j < 3; j++){
				fprintf(stdout, "   c-%d %d %d IOU: %.2f%% TPIOU: %.2f%% Recall: %.2f%% Precision: %.2f%%\n", j, clscorrect[j], clsproposals[j], clsaveiou[j] * 100 / clstruth[j], clstpiou[j] * 100 / clscorrect[j], 100.*clscorrect[j] / clstruth[j], 100.*clscorrect[j] / clsproposals[j]);
			}
			fprintf(stdout, "   %d %d %d RPs/Img: %.2f IOU: %.2f%% TPIOU: %.2f%% Recall: %.2f%% Precision: %.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, tp_iou * 100 / correct, 100.*correct / total, 100.*correct / proposals);
		}
		else {
			fprintf(stdout, "   %d %d %d RPs/Img: %.2f IOU: %.2f%% TPIOU: %.2f%% Recall: %.2f%% Precision: %.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 0.0, 0.0, 0.0);
		}
		free(id);
		free_image(orig);
		free_image(sized);
	}
}

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n-1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
        if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
        save_image(im, "predictions");
        show_image(im, "predictions");

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

box get_region_box2(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
	box b;
	b.x = (i + logistic_activate(x[index + 0])) / w;
	b.y = (j + logistic_activate(x[index + 1])) / h;
	b.w = exp(x[index + 2]) * biases[2 * n] / w;
	b.h = exp(x[index + 3]) * biases[2 * n + 1] / h;
	return b;
}

void one_hot(network net, network_state state, int clsid)
{
	int i, j, n, b, t;
	layer l = net.layers[net.n - 1];
	float *predictions = l.output;
	int size = l.coords + l.classes + 1;

	int locations = l.side*l.side;
	for (i = 0; i < l.w*l.h; ++i){
		for (n = 0; n < l.n; ++n){
			int index = i*l.n + n;
			int p_index = index * (l.classes + 5) + 4;
			float scale = predictions[p_index];
			int class_index = index * (l.classes + 5) + 5;
			for (j = 0; j < l.classes; ++j){
				if (clsid == j){
					predictions[class_index + j] = 1.0f / scale;
				}
				else{
					predictions[class_index + j] = 0;
				}
			}
		}
	}

	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
	float avg_iou = 0;
	float recall = 0;
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;

	for (b = 0; b < l.batch; ++b) {
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				for (n = 0; n < l.n; ++n) {
					int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
					box pred = get_region_box2(l.output, l.biases, n, index, i, j, l.w, l.h);
					float best_iou = 0;
					for (t = 0; t < 30; ++t){
						box truth = float_to_box(state.truth + t * 5 + b*l.truths);
						if (!truth.x) break;
						float iou = box_iou(pred, truth);
						if (iou > best_iou) {
							best_iou = iou;
						}
					}
					avg_anyobj += l.output[index + 4];
					l.delta[index + 4] = l.noobject_scale * ((0 - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
					if (best_iou > l.thresh) {
						l.delta[index + 4] = 0;
					}

			//		if (*(state.net.seen) < 12800){
			//			box truth = { 0 };
			//			truth.x = (i + .5) / l.w;
			//			truth.y = (j + .5) / l.h;
			//			truth.w = l.biases[2 * n] / l.w;
			//			truth.h = l.biases[2 * n + 1] / l.h;
			//			delta_region_box(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
			//		}
				}
			}
		}
	}


	if (gpu_index > 0){
		cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
	}
}

image grad_image(network net)
{
	layer l = net.layers[8];
	image im = make_image(416, 416, 1);
	int size, i, j;
	float *out, *grads, *weights, *cam, maxfind;
	size = l.h*l.w*l.n;
	out = calloc(size, sizeof(float));
	grads = calloc(size, sizeof(float));
	cam = calloc(size, sizeof(float));
	if (gpu_index > 0){
		cuda_pull_array(l.output_gpu, out, size);
		cuda_pull_array(l.delta_gpu, grads, size);
		cuda_pull_array(l.mean_delta_gpu, grads, l.n);
	}
	else{
		out = l.output;
		grads = l.mean_delta;
	}
	for (i = 0; i < l.n; i++){
		for (j = 0; j < l.h*l.w; j++){
			int index = i*l.h*l.w + j;
			cam[index] = out[index] *grads[i];
		}
	}
	maxfind = 0;
	for (j = 0; j < l.h*l.w*l.n; j++){
		if (cam[j] > maxfind){
			maxfind = cam[j];
		}
	}
	for (j = 0; j < l.h*l.w*l.n; j++){
		cam[j] = fmax(cam[j], 0) / maxfind * 255;
	}


	memcpy(im.data, cam, size > 416 * 416 ? 416 * 416 * sizeof(float) : size * sizeof(float));
	show_image(im, "cam");
	return im;
}

float* take_output(network net)
{
	layer l = net.layers[8];
	int size = l.h*l.w*l.n;
	float* out = calloc(size, sizeof(float));
	if (gpu_index > 0){
		cuda_pull_array(l.output_gpu, out, size);
	}
	else{
		memcpy(out, l.output, size * sizeof(float));
	}
	return out;
}



void grad_cam_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
{
	list *options = read_data_cfg(datacfg);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);

	image **alphabet = load_alphabet();
	network net = parse_network_cfg(cfgfile);
	if (weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms = .4;
	while (1){
		if (filename){
			strncpy(input, filename, 256);
		}
		else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input) return;
			strtok(input, "\n");
		}
		image im = load_image_color(input, 0, 0);
		image sized = resize_image(im, net.w, net.h);
		layer l = net.layers[net.n - 1];

		box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
		float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

		float *X = sized.data;
		float *out;
		time = clock();


		network_state state;
		state.net = net;
		state.index = 0;
		state.truth = calloc(30, sizeof(float));
		state.train = 1;
		state.delta = 0;
		state.truth[0] = 0.674556;
		state.truth[1] = 0.707407;
		state.truth[2] = 0.088757;
		state.truth[3] = 0.111111;
		gpu_index = -1;
		if (gpu_index > 0){
			cuda_set_device(net.gpu_index);
			int size = get_network_input_size(net) * net.batch;
			state.input = cuda_make_array(X, size);
			network_predict(net, X);
		}else{
			state.net = net;
			state.input = X;
			network_predict(net, X);
//			out = get_network_output(net);
		}

//		take_output(net);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
		get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
		if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
		else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
		draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
		save_image(im, "predictions");
		show_image(im, "predictions");

		one_hot(net, state, 1);
		if (gpu_index > 0){
			backward_network_gpu(net, state);
		}else{
			backward_network(net, state);
		}
		image gim = grad_image(net);
		free_image(im);
		free_image(sized);
		free(boxes);
		free_ptrs((void **)probs, l.w*l.h*l.n);
#ifdef OPENCV
		cvWaitKey(0);
		cvDestroyAllWindows();
#endif
		if (filename) break;
	}
}


void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .24);
	float iouth = find_float_arg(argc, argv, "-iouth", .5);
	float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
	float nms = find_float_arg(argc, argv, "-nms", 0.4);
	int stimg = find_int_arg(argc, argv, "-stimg", 0);
	int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
	if (argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
	else if (0 == strcmp(argv[2], "recall")) validate_detector_recall1(datacfg, cfg, weights, thresh, iouth, nms, stimg);
	else if (0 == strcmp(argv[2], "recall2")) validate_detector_recall2(datacfg, cfg, weights, thresh, iouth, nms, stimg);
	else if (0 == strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
		demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, hier_thresh, outfile);
    }
}

