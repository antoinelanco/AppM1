#ifndef GATHER_DATA_H
#define GATHER_DATA_H

#include "data.h"
#include "../model/k_means.h"
#include <vector>
#include <tuple>

using namespace std;

/**
 *  Retourne des data de 4 * N fetures déduite de k_means_patch
 */
vector<data> gatherDataFeatures(K_means k_means_patch, vector<data> splittedData, int N);

/**
 *  Retourne des data de 4 * N features et le K_means appris sur ces données
 */
pair<vector<data>, K_means > trainKMeansDataFeatures(vector<data> splittedData, int N, int nbIter);

#endif
