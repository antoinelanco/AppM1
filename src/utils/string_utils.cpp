//
// Created by samuel on 03/02/18.
//

#include "string_utils.h"
#include <sstream>

vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> elems;
    while (getline(ss, item, delim)) {
        //elems.push_back(item);
        elems.push_back(move(item)); // if C++11 (based on comment from @mchiasson)
    }
    return elems;
}
