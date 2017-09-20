#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include <cmath>
#include "nn_params.h"

void map_feature(const int16_t raw[][3]) {
	inputs[0]=raw[0][0];inputs[1]=raw[0][1];inputs[2]=raw[0][2];
	inputs[3]=raw[0][0];inputs[4]=raw[0][1];inputs[5]=raw[0][2];
	inputs[6]=raw[0][0];inputs[7]=raw[0][1];inputs[8]=raw[0][2];
	for(short c=0; c<3; c++) {
		for(short r=1; r<buff_size; r++) {
			if(inputs[c]<raw[r][c])
				inputs[c] = raw[r][c];
			if(inputs[c+3]>raw[r][c])
				inputs[c+3] = raw[r][c];
			inputs[c+6] += raw[r][c];
		}
		inputs[c+6] /= 35.0;
		inputs[c+9]=0;
		for(short r=0; r<buff_size; r++) {
			inputs[c+9] += (raw[r][c]-inputs[c+6])*(raw[r][c]-inputs[c+6]);
		}
		inputs[c+9] = sqrt(inputs[c+9]/33.0);
	}
};

void normalize() {
	for(short i=0; i<input_size; i++)
		inputs[i] = float(inputs[i]-normal_mean[i]) / normal_std[i];
};

void sigmoid(float & z) {
	z = 1.0 / (1.0 + exp(-z));
};

void hidden_activate() {
	for(short i=0; i<hidden_size; i++) {
		hidden_activ[i] = weight1[i][0];
		for(short j=1; j<=input_size; j++)
			hidden_activ[i] += weight1[i][j] * inputs[j-1];
		sigmoid(hidden_activ[i]);
	}
};

void output_activate() {
	for(short i=0; i<num_labels; i++) {
		output_activ[i] = weight2[i][0];
		for(short j=1; j<=hidden_size; j++)
			output_activ[i] += weight2[i][j] * hidden_activ[j-1];
		sigmoid(output_activ[i]);
	}
};

int map_label() {
	short label = 0; float maxl = output_activ[0];
	for(short i=1; i<num_labels; i++) {
		if(maxl<output_activ[i]) {
			maxl = output_activ[i]; label = i;
		}
	}
	return (label+1);
};

int classify(const int16_t raw[][3]) {
	map_feature(raw);
	normalize();
	hidden_activate();
	output_activate();
	return map_label();
};

#endif
