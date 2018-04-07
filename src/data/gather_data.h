#ifndef GATHER_DATA_H
#define GATHER_DATA_H

#include "data.h"
#include "../model/k_means.h"
#include <vector>
#include <tuple>

using namespace std;

/**
 *  Retourne des data de nbPatch * N fetures déduite de k_means_patch
 */
vector<data> gatherDataFeatures(K_Means_2 k_means_patch, vector<data> splittedData, int N, int nbPatch);

/**
 *  Retourne des data de nbPatch * N features et le K_means appris sur ces données
 */
pair<vector<data>, K_means > trainKMeansDataFeatures(vector<data> splittedData, int N, int nbIter, int nbPatch);

#endif
