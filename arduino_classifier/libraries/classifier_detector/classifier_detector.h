#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <cmath>
#include "move_params.h"
#include "start_params.h"
#include "end_params.h"

void map_feature_move(const int16_t raw[][3]) {
	inputs_move[0]=raw[0][0];inputs_move[1]=raw[0][1];inputs_move[2]=raw[0][2];
	inputs_move[3]=raw[0][0];inputs_move[4]=raw[0][1];inputs_move[5]=raw[0][2];
	inputs_move[6]=raw[0][0];inputs_move[7]=raw[0][1];inputs_move[8]=raw[0][2];

	for(short c=0; c<3; c++) {
		for(short r=1; r<count; r++) {
			if(inputs_move[c]<raw[r][c])
				inputs_move[c] = raw[r][c];
			if(inputs_move[c+3]>raw[r][c])
				inputs_move[c+3] = raw[r][c];
			inputs_move[c+6] += raw[r][c];
		}
		inputs_move[c+6] /= count;
		inputs_move[c+9]=0;
		for(short r=0; r<count; r++) {
			inputs_move[c+9] += (raw[r][c]-inputs_move[c+6])*(raw[r][c]-inputs_move[c+6]);
		}
		inputs_move[c+9] = sqrt((float(inputs_move[c+9]))/(count-1));
	}
	for(short i=0; i<input_size_move; i++)
		inputs_move[i] = float(inputs_move[i]-normal_mean_move[i]) / normal_std_move[i];
};

void map_feature_start(const int16_t raw[][3]) {
	for(short r=0; r<buff_size_start; r++) {
		inputs_start[r*3] = float(raw[r][0]-normal_mean_start[r*3]) / normal_std_start[r*3];
		inputs_start[r*3+1] = float(raw[r][1]-normal_mean_start[r*3+1]) / normal_std_start[r*3+1];
		inputs_start[r*3+2] = float(raw[r][2]-normal_mean_start[r*3+2]) / normal_std_start[r*3+2];
	}
};

void map_feature_end(const int16_t raw[][3]) {
	uint8_t start = count - buff_size_end;
	for(short r=0; r<buff_size_end; r++) {
		inputs_end[r*3] = float(raw[r+start][0]-normal_mean_end[r*3]) / normal_std_end[r*3];
		inputs_end[r*3+1] = float(raw[r+start][1]-normal_mean_end[r*3+1]) / normal_std_end[r*3+1];
		inputs_end[r*3+2] = float(raw[r+start][2]-normal_mean_end[r*3+2]) / normal_std_end[r*3+2];
	}
};

void map_feature(const int16_t raw[][3], const uint8_t & _t) {
	switch(_t) {
		case 0: map_feature_move(raw); break;
		case 1: map_feature_start(raw); break;
		case 2: map_feature_end(raw);
	}
};

void sigmoid(float & z) {
	z = 1.0 / (1.0 + exp(-z));
};

void activate_move() {
	for(short i=0; i<hidden_size_move; i++) {
		hidden_activ_move[i] = weight1_move[i][0];
		for(short j=1; j<=input_size_move; j++)
			hidden_activ_move[i] += weight1_move[i][j] * inputs_move[j-1];
		sigmoid(hidden_activ_move[i]);
	}
	for(short i=0; i<output_size_move; i++) {
		output_activ_move[i] = weight2_move[i][0];
		for(short j=1; j<=hidden_size_move; j++)
			output_activ_move[i] += weight2_move[i][j] * hidden_activ_move[j-1];
		sigmoid(output_activ_move[i]);
	}
};

void activate_start() {
	for(short i=0; i<hidden_size_start; i++) {
		hidden_activ_start[i] = weight1_start[i][0];
		for(short j=1; j<=input_size_start; j++)
			hidden_activ_start[i] += weight1_start[i][j] * inputs_start[j-1];
		sigmoid(hidden_activ_start[i]);
	}
	for(short i=0; i<output_size_start; i++) {
		output_activ_start[i] = weight2_start[i][0];
		for(short j=1; j<=hidden_size_start; j++)
			output_activ_start[i] += weight2_start[i][j] * hidden_activ_start[j-1];
		sigmoid(output_activ_start[i]);
	}
};

void activate_end() {
	for(short i=0; i<hidden_size_end; i++) {
		hidden_activ_end[i] = weight1_end[i][0];
		for(short j=1; j<=input_size_end; j++)
			hidden_activ_end[i] += weight1_end[i][j] * inputs_end[j-1];
		sigmoid(hidden_activ_end[i]);
	}
	for(short i=0; i<output_size_end; i++) {
		output_activ_end[i] = weight2_end[i][0];
		for(short j=1; j<=hidden_size_end; j++)
			output_activ_end[i] += weight2_end[i][j] * hidden_activ_end[j-1];
		sigmoid(output_activ_end[i]);
	}
};

void activate(const uint8_t & _t) {
	switch(_t) {
		case 0: activate_move(); break;
		case 1: activate_start(); break;
		case 2: activate_end();
	}
};


int map_label(float output_activ[], const uint8_t & _t) {
	uint8_t num_labels = (_t==0) ? 4 : 2;
	short label = 0; float maxl = output_activ[0];
	for(short i=1; i<num_labels; i++) {
		if(maxl<output_activ[i]) {
			maxl = output_activ[i]; label = i;
		}
	}
	return (label+1);
};

int classify(const int16_t raw[][3], const uint8_t & type) {
	map_feature(raw, type);
	activate(type);
	uint8_t result;
	if(type==0)
		result = map_label(output_activ_move, type);
	else if(type==1)
		result = map_label(output_activ_start, type);
	else
		result = map_label(output_activ_end, type);

	return result;
};

#endif
